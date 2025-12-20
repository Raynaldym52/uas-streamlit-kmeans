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
    df = pd.read_csv("BencanaPWK2022.csv")
    return df

df = load_data()

# ===============================
# Normalisasi Nama Jenis Bencana
# (agar konsisten dengan yang diminta)
# ===============================
df['Jenis_Bencana'] = df['Jenis_Bencana'].str.lower()

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
    st.write("Data bencana alam berdasarkan wilayah dan jenis bencana.")
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

    st.write("### Statistik Deskriptif")
    st.write(df.describe(include='all'))

    # Jumlah kejadian per jenis bencana
    st.write("### Jumlah Kejadian per Jenis Bencana")
    st.dataframe(df['Jenis_Bencana'].value_counts())

# ===============================
# Menu Visualisasi
# ===============================
elif menu == "Visualisasi":
    st.subheader("📉 Visualisasi Data Bencana")

    # Daftar jenis bencana yang diminta
    daftar_bencana = [
        "pohon tumbang",
        "angin puting beliung",
        "longsor tanah",
        "gempa",
        "karhutla",
        "bangunan ambruk",
        "kekeringan"
    ]

    # Filter hanya data yang sesuai daftar
    df_visual = df[df['Jenis_Bencana'].isin(daftar_bencana)]

    jenis = st.selectbox(
        "Pilih Jenis Bencana",
        sorted(df_visual['Jenis_Bencana'].unique())
    )

    df_filter = df_visual[df_visual['Jenis_Bencana'] == jenis]

    st.write(f"Menampilkan data untuk **{jenis.upper()}**")

    # Grafik jumlah kejadian per wilayah
    fig, ax = plt.subplots(figsize=(10, 5))
    df_filter['Wilayah'].value_counts().plot(
        kind='bar',
        ax=ax
    )
    ax.set_title(f"Jumlah Kejadian {jenis} per Wilayah")
    ax.set_xlabel("Wilayah")
    ax.set_ylabel("Jumlah Kejadian")
    plt.xticks(rotation=45)

    st.pyplot(fig)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown("📌 **UAS Data Mining – Aplikasi Streamlit Bencana Alam**")
