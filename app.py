import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="UAS Data Mining - KMeans",
    layout="wide"
)

# ===============================
# Load Data & Model
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("BencanaPWK2022.csv")

@st.cache_resource
def load_model():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

df = load_data()
kmeans_model, scaler = load_model()

# ===============================
# Judul
# ===============================
st.title("📊 Aplikasi Clustering Data Bencana (K-Means)")
st.write(
    """
    Aplikasi ini merupakan implementasi **K-Means Clustering**
    untuk mengelompokkan data bencana menggunakan model yang
    telah dilatih sebelumnya.
    """
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
# Menu Clustering
# ===============================
elif menu == "Clustering":
    st.subheader("🔍 Hasil Clustering K-Means")

    # Ambil kolom numerik
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    st.write("Kolom numerik yang digunakan:")
    st.write(list(numeric_cols))

    # Scaling
    X = df[numeric_cols]
    X_scaled = scaler.transform(X)

    # Prediksi cluster
    cluster_result = kmeans_model.predict(X_scaled)
    df["Cluster"] = cluster_result

    st.subheader("📌 Data dengan Label Cluster")
    st.dataframe(df)

    # Visualisasi
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Pilih Sumbu X", numeric_cols)
        y_axis = st.selectbox("Pilih Sumbu Y", numeric_cols)

        fig, ax = plt.subplots()
        sns.scatterplot(
            x=df[x_axis],
            y=df[y_axis],
            hue=df["Cluster"],
            palette="Set1",
            ax=ax
        )

        ax.set_title("Visualisasi Clustering K-Means")
        st.pyplot(fig)
    else:
        st.warning("Minimal dibutuhkan 2 kolom numerik untuk visualisasi.")
