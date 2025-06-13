import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
import re
# from PIL import Image

# --- Konfigurasi Halaman (HARUS DI BAGIAN PALING ATAS!) ---
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="centered", page_icon="🚗")

# Custom CSS untuk tampilan lebih menarik (juga bisa diletakkan di sini)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: #f0f2f6; /* Warna latar belakang yang lebih terang */
        padding: 20px;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #28a745; /* Warna hijau untuk tombol utama */
        color: white;
        padding: 0.8em 1.5em;
        border-radius: 10px;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #218838;
        box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .stSelectbox, .stNumberInput, .stTextInput, .stSlider {
        background-color: #ffffff;
        border-radius: 12px; /* Radius lebih besar */
        padding: 0.8em 1.2em; /* Padding lebih sedikit */
        margin-bottom: 1em;
        box-shadow: 1px 1px 4px rgba(0,0,0,0.1); /* Sedikit shadow */
        border: 1px solid #e0e0e0; /* Border halus */
    }
    h1 {
        color: #004d80; /* Warna judul lebih gelap */
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.5em;
    }
    h2, h3, h4, h5, h6 {
        color: #0056b3;
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }
    .stMarkdown {
        text-align: center;
        color: #555;
    }
    .stAlert {
        border-radius: 10px;
    }
    .stAlert.info {
        background-color: #e0f2f7; /* Warna latar belakang info yang lebih lembut */
        border-left: 8px solid #007bff; /* Garis samping biru */
        color: #333;
    }
    .stAlert.success {
        background-color: #e6ffe6;
        border-left: 8px solid #28a745;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)


# --- Load Model ---
# Menggunakan st.cache_resource agar model hanya dimuat sekali
@st.cache_resource
def load_model():
    try:
        with open("catboost_model_quikr.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: 'catboost_model_quikr.pkl' not found. Pastikan file model ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop()

model = load_model()

# --- Load and Preprocess Data for Filtering ---
# Menggunakan st.cache_data agar data hanya dimuat dan diproses sekali
@st.cache_data
def load_filter_data(file_path="mobilbekas.csv"):
    try:
        df = pd.read_csv(file_path)

        # Basic cleaning for relevant columns to ensure filtering works
        if 'harga' in df.columns:
            df['harga'] = pd.to_numeric(df['harga'], errors='coerce')
        if 'tahun' in df.columns:
            df['tahun'] = pd.to_numeric(df['tahun'], errors='coerce')
        if 'jarak_tempuh' in df.columns:
            def parse_kms(km_str):
                if pd.isna(km_str): return np.nan
                km_str = str(km_str).replace(".", "").replace(",", "").strip()
                digits = re.findall(r'\d+', km_str)
                if digits:
                    if '-' in km_str:
                        parts = [int(p) for p in digits]
                        return (int(parts[0]) + int(parts[1])) / 2 if len(parts) >= 2 else int(digits[0])
                    return int(digits[0])
                return np.nan
            df['jarak_tempuh'] = df['jarak_tempuh'].apply(parse_kms)

        for col in ['merek', 'model', 'tipe_bahan_bakar', 'transmisi', 'warna', 'varian']:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown')
            else:
                df[col] = 'Unknown'

        df.dropna(subset=['merek', 'model', 'tipe_bahan_bakar', 'transmisi'], inplace=True)
        return df

    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. File ini dibutuhkan untuk filtering dinamis. Pastikan ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau memproses dataset untuk filtering: {e}")
        st.stop()

df_filter_data = load_filter_data()

# Tambahkan gambar (opsional)
# try:
#     image = Image.open('car_banner.png') # Ganti dengan nama file gambar Anda
#     st.image(image, use_column_width=True)
# except FileNotFoundError:
#     st.caption("Tambahkan 'car_banner.png' di direktori yang sama untuk gambar header.")

st.title("🚗 Prediksi Harga Mobil Bekas")
st.markdown("Gunakan aplikasi ini untuk memprediksi harga mobil bekas berdasarkan data kendaraan Anda.")

st.subheader("Input Detail Mobil")

# --- Input Pengguna dengan Filter Berurutan dan Layout Kolom ---

# Baris 1: Merek dan Model
col1, col2 = st.columns(2)
with col1:
    unique_merek = sorted(df_filter_data['merek'].unique().tolist())
    company_input = st.selectbox("Pilih Merek / Brand", unique_merek, key="merek_select")

with col2:
    filtered_by_merek = df_filter_data[df_filter_data['merek'] == company_input]
    unique_model = sorted(filtered_by_merek['model'].unique().tolist())
    if not unique_model:
        unique_model = ['Tidak Ada Model Ditemukan']
    name_input = st.selectbox("Pilih Model Mobil", unique_model, key="model_select")

# Baris 2: Bahan Bakar dan Transmisi
col3, col4 = st.columns(2)
with col3:
    filtered_by_model_for_fuel = filtered_by_merek[filtered_by_merek['model'] == name_input]
    unique_fuel_types = sorted(filtered_by_model_for_fuel['tipe_bahan_bakar'].unique().tolist())
    if not unique_fuel_types:
        unique_fuel_types = ['Tidak Ada']
    fuel_type_input = st.selectbox("Pilih Tipe Bahan Bakar", unique_fuel_types, key="fuel_select")

with col4:
    filtered_by_model_for_trans = filtered_by_merek[filtered_by_merek['model'] == name_input]
    unique_transmission_types = sorted(filtered_by_model_for_trans['transmisi'].unique().tolist())
    if not unique_transmission_types:
        unique_transmission_types = ['Tidak Ada']
    transmission_type_input = st.selectbox("Pilih Tipe Transmisi", unique_transmission_types, key="transmisi_select")

# Baris 3: Warna dan Varian
col5, col6 = st.columns(2)
with col5:
    common_colors = sorted(df_filter_data['warna'].astype(str).unique().tolist())
    color_input = st.selectbox("Warna Mobil", common_colors, key="color_select")

with col6:
    common_variants = sorted(df_filter_data['varian'].astype(str).unique().tolist())
    variant_input = st.selectbox("Varian Mobil", common_variants, key="varian_select")

# Baris 4: Umur dan Jarak Tempuh
st.markdown("---") # Garis pemisah sebelum input numerik
col7, col8 = st.columns(2)
with col7:
    age_input = st.slider("Umur Mobil (tahun)", 0, 30, 5, key="age_slider")
with col8:
    kms_driven_input = st.number_input("Jarak Tempuh (dalam KM)", min_value=0, step=1000, key="kms_input")


# --- Feature Engineering to match the model's training script ---
# These calculations must mirror the training script exactly.

# company_model
company_model_feature = f"{company_input}_{name_input}"

# segment (based on estimated price, as actual price is unknown for prediction)
base_prices = {
    'Audi A6': 650_000_000, 'BMW 5 Series': 700_000_000, 'Mercedes C-Class': 750_000_000,
    'Toyota Fortuner': 350_000_000, 'Honda City': 300_000_000, 'Hyundai i20': 250_000_000,
    'Maruti Swift': 120_000_000, 'Tata Nano': 110_000_000, 'Daihatsu Ayla': 130_000_000,
    'Wuling Confero': 150_000_000, 'Hyundai Creta': 320_000_000, 'Suzuki Karimun': 90_000_000,
    'DFSK Glory': 200_000_000, 'Volkswagen Polo': 280_000_000, 'Mazda CX-5': 450_000_000
}
estimated_price = base_prices.get(name_input, 200_000_000) # Default if car not in list
segment_bins = [0, 80e6, 150e6, 300e6, 500e6, 1e9, 3e9]
segment_labels = ['ultra_low', 'low', 'mid_low', 'mid_high', 'high', 'lux']
segment_feature = pd.cut([estimated_price], bins=segment_bins, labels=segment_labels, right=False)[0]


# fuel_age
fuel_age_feature = f"{fuel_type_input}_{age_input}"

# company_segment
company_segment_feature = f"{company_model_feature}_{segment_feature}"

# log_km
log_km_feature = np.log1p(kms_driven_input)

# log_km_per_year
log_km_per_year_feature = np.log1p(kms_driven_input / max(age_input, 1))

# brand_category (based on model name)
lux_names = ['BMW 5 Series', 'Mercedes C-Class', 'Audi A6', 'Land Rover Evoque']
mid_names = ['Toyota Fortuner', 'Honda City', 'Hyundai Creta', 'Volkswagen Polo', 'Mazda CX-5']
budget_names = ['Daihatsu Ayla', 'Suzuki Karimun', 'Wuling Confero', 'Tata Nano', 'Chery QQ', 'DFSK Glory']

if name_input in lux_names:
    brand_category_feature = 'luxury'
elif name_input in mid_names:
    brand_category_feature = 'midrange'
elif name_input in budget_names:
    brand_category_feature = 'budget'
else:
    brand_category_feature = 'general'

# Flags tambahan
is_premium_feature = int(brand_category_feature == 'luxury' and age_input <= 3)
is_high_value_feature = int(brand_category_feature in ['luxury', 'midrange'] and age_input <= 5 and kms_driven_input <= 60000)
is_low_budget_feature = int(brand_category_feature == 'budget' and age_input >= 10 and kms_driven_input >= 120000)

# Negative age feature (for monotonicity if implemented in training)
negative_age_feature = -age_input # Calculate as per previous discussion


# DataFrame input for prediction (ensure column names and order match model training)
features_df = pd.DataFrame([{
    'company_model': company_model_feature,
    'tipe_bahan_bakar': fuel_type_input,
    'log_km': log_km_feature,
    'negative_age': negative_age_feature, # Menggunakan negative_age_feature jika model dilatih dengan ini
    'segment': segment_feature,
    'fuel_age': fuel_age_feature,
    'company_segment': company_segment_feature,
    'log_km_per_year': log_km_per_year_feature,
    'brand_category': brand_category_feature,
    'is_premium': is_premium_feature,
    'is_high_value': is_high_value_feature,
    'is_low_budget': is_low_budget_feature,
    'transmisi': transmission_type_input,
    'warna': color_input,
    'varian': variant_input
}])

# Prediksi harga
if st.button("Prediksi Harga"):
    try:
        pred_log = model.predict(features_df)[0]
        pred_rp = np.expm1(pred_log)
        st.subheader(f"💰 Estimasi Harga: Rp {pred_rp:,.0f}")
        st.success("Prediksi berhasil!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.info("Pastikan semua input sudah diisi dengan benar dan model sudah dimuat.")

# --- Bagian Performa Model Dipindahkan ke Bawah ---
st.markdown("---") # Garis pemisah tambahan
st.write("") # Spasi kosong

# Menggunakan columns untuk menempatkan di kiri bawah
col_metrics, _ = st.columns([1, 2]) # Kolom kiri untuk metrik, kolom kanan kosong
with col_metrics:
    st.subheader("📊 Performa Model")
    st.info(f"""
        **Akurasi Model Berdasarkan Data Uji:**
        * **MAE (Mean Absolute Error):** Rp 22.263.400
        * **RMSE (Root Mean Squared Error):** Rp 44.287.777
        * **R² (R-squared):** 0.964
        
        Nilai R² yang tinggi (mendekati 1) menunjukkan bahwa model dapat menjelaskan sebagian besar variasi harga mobil.
    """)
