import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="UAS KELOMPOK 9- KMeans",
    layout="wide"
)

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("BencanaPWK2022.csv")

df = load_data()

# ===============================
# Judul
# ===============================
st.title("📊 Aplikasi Clustering Data Bencana (K-Means)")
st.write(
    "Aplikasi ini melakukan **clustering K-Means langsung dari dataset** "
    "tanpa menggunakan model eksternal."
)

# ===============================
# Sidebar Menu
# ===============================
st.sidebar.header("Menu")
menu = st.sidebar.selectbox(
    "Pilih Menu",
    ["Dataset", "Statistik", "Clustering"]
)

# ===============================
# Menu Dataset
# ===============================
if menu == "Dataset":
    st.subheader("📁 Dataset Bencana")
    st.write("Jumlah Baris:", df.shape[0])
    st.write("Jumlah Kolom:", df.shape[1])
    st.dataframe(df)

# ===============================
# Menu Statistik
# ===============================
elif menu == "Statistik":
    st.subheader("📈 Statistik Deskriptif")
    st.write(df.describe())

# ===============================
# Menu Clustering (PASTI JALAN)
# ===============================
elif menu == "Clustering":
    st.subheader("🔍 Hasil Clustering K-Means")

    # Fitur numerik sesuai CSV
    fitur = [
        "pohon tumbang",
        "banjir",
        "longsor tanah",
        "Jumlah Bencana"
    ]

    st.write("Fitur yang digunakan:")
    st.write(fitur)

    # Validasi kolom
    missing = [f for f in fitur if f not in df.columns]
    if missing:
        st.error(f"Kolom tidak ditemukan: {missing}")
        st.stop()

    # Ambil & bersihkan data
    X = df[fitur]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    # Input jumlah cluster
    k = st.slider("Jumlah Cluster (K)", 2, 5, 3)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    st.subheader("📌 Data dengan Label Cluster")
    st.dataframe(df)

    # ===============================
    # Visualisasi (AMAN)
    # ===============================
    x_axis = st.selectbox("Pilih Sumbu X", fitur)
    y_axis = st.selectbox("Pilih Sumbu Y", fitur)

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df[x_axis],
        df[y_axis],
        c=df["Cluster"]
    )

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title("Visualisasi Hasil Clustering K-Means")

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster"
    )
    ax.add_artist(legend)

    st.pyplot(fig)
