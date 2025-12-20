import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Aplikasi Bencana Alam",
    page_icon="🌋",
    layout="wide"
)

# ===============================
# Judul Aplikasi
# ===============================
st.title("🌋 Aplikasi Informasi Bencana Alam")
st.write("Aplikasi ini digunakan untuk menampilkan dan menganalisis data bencana alam.")

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("BencanaPWK2022.csv")

df = load_data()

# ===============================
# TAMPILKAN NAMA KOLOM (DEBUG AMAN)
# ===============================
st.sidebar.write("Kolom pada dataset:")
st.sidebar.write(list(df.columns))

# ===============================
# PENYESUAIAN NAMA KOLOM (ANTI ERROR)
# ===============================
# Ganti nama kolom sesuai CSV
if 'Jenis_Bencana' not in df.columns:
    for col in df.columns:
        if 'bencana' in col.lower():
            df.rename(columns={col: 'Jenis_Bencana'}, inplace=True)

if 'Wilayah' not in df.columns:
    for col in df.columns:
        if 'wilayah' in col.lower() or 'kabupaten' in col.lower() or 'kota' in col.lower():
            df.rename(columns={col: 'Wilayah'}, inplace=True)

# Normalisasi teks
df['Jenis_Bencana'] = df['Jenis_Bencana'].astype(str).str.lower()
df['Wilayah'] = df['Wilayah'].astype(str)

# ===============================
# Sidebar
# ===============================
st.sidebar.header("Menu")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Dataset", "Statistik", "Visualisasi"]
)

# ===============================
# Menu Dataset
# ===============================
if menu == "Dataset":
    st.subheader("📊 Dataset Bencana Alam")
    st.dataframe(df)

# ===============================
# Menu Statistik
# ===============================
elif menu == "Statistik":
    st.subheader("📈 Statistik Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Data", len(df))

    with col2:
        st.metric("Jumlah Jenis Bencana", df['Jenis_Bencana'].nunique())

    with col3:
        st.metric("Jumlah Wilayah", df['Wilayah'].nunique())

    st.write("### Jumlah Kejadian per Jenis Bencana")
    st.dataframe(df['Jenis_Bencana'].value_counts())

# ===============================
# Menu Visualisasi
# ===============================
elif menu == "Visualisasi":
    st.subheader("📉 Visualisasi Data Bencana")

    daftar_bencana = [
        "pohon tumbang",
        "angin puting beliung",
        "longsor tanah",
        "gempa",
        "karhutla",
        "bangunan ambruk",
        "kekeringan"
    ]

    df_visual = df[df['Jenis_Bencana'].isin(daftar_bencana)]

    jenis = st.selectbox(
        "Pilih Jenis Bencana",
        sorted(df_visual['Jenis_Bencana'].unique())
    )

    df_filter = df_visual[df_visual['Jenis_Bencana'] == jenis]

    fig, ax = plt.subplots(figsize=(10, 5))
    df_filter['Wilayah'].value_counts().plot(kind='bar', ax=ax)

    ax.set_title(f"Jumlah Kejadian {jenis}")
    ax.set_xlabel("Wilayah")
    ax.set_ylabel("Jumlah Kejadian")
    plt.xticks(rotation=45)

    st.pyplot(fig)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown("📌 **UAS Data Mining – Aplikasi Streamlit Bencana Alam**")
