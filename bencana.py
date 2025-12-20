import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ===============================
# FONT CONFIG (DejaVu Sans)
# ===============================
plt.rcParams["font.family"] = "DejaVu Sans"

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Clustering Bencana Alam",
    page_icon="🌋",
    layout="wide"
)

# ===============================
# JENIS BENCANA
# ===============================
JENIS_BENCANA = [
    'pohon tumbang',
    'angin_puting_beliung',
    'longsor tanah',
    'banjir',
    'gempa',
    'karhutla',
    'bangunan ambruk',
    'kekeringan'
]

# ===============================
# JUDUL
# ===============================
st.title("🌋 Aplikasi Clustering Bencana Alam")
st.markdown(
    "Aplikasi ini digunakan untuk **mengelompokkan data bencana alam** "
    "menggunakan algoritma **K-Means Clustering**."
)

# ===============================
# LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("BencanaPWK2022.csv")

@st.cache_resource
def load_model():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

df_full = load_data()
kmeans_model, scaler = load_model()

# ===============================
# SIDEBAR
# ===============================
menu = st.sidebar.radio(
    "📌 Menu",
    ["Data", "Visualisasi Cluster", "Evaluasi Model", "Prediksi Cluster"]
)

st.sidebar.markdown("### 🔎 Filter Jenis Bencana")

selected_bencana = st.sidebar.multiselect(
    "Pilih Jenis Bencana",
    JENIS_BENCANA,
    default=JENIS_BENCANA
)

# ===============================
# FILTER DATA (AMAN - FIX ERROR LINE 123)
# ===============================
if 'jenis_bencana' in df_full.columns:
    df = df_full[df_full['jenis_bencana'].isin(selected_bencana)]
else:
    df = df_full.copy()

if df.empty:
    st.warning("⚠️ Data kosong setelah filter. Silakan pilih jenis bencana.")
    st.stop()

# ===============================
# MENU : DATA
# ===============================
if menu == "Data":
    st.subheader("📊 Dataset Bencana")
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    if 'jenis_bencana' in df.columns:
        st.subheader("📌 Distribusi Jenis Bencana")
        st.bar_chart(df['jenis_bencana'].value_counts())

# ===============================
# MENU : VISUALISASI
# ===============================
elif menu == "Visualisasi Cluster":
    st.subheader("📉 Visualisasi Clustering (PCA 2D)")

    X = df.select_dtypes(include=["int64", "float64"])
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame({
        "PCA1": pca_result[:, 0],
        "PCA2": pca_result[:, 1],
        "Cluster": kmeans_model.predict(X_scaled)
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_pca,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette="Set2",
        s=80,
        ax=ax
    )

    ax.set_title("Visualisasi Cluster Data Bencana", fontsize=14)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    st.pyplot(fig)

# ===============================
# MENU : EVALUASI
# ===============================
elif menu == "Evaluasi Model":
    st.subheader("📊 Evaluasi Model K-Means")

    X = df.select_dtypes(include=["int64", "float64"])
    X_scaled = scaler.transform(X)

    sil = silhouette_score(X_scaled, kmeans_model.predict(X_scaled))
    dbi = davies_bouldin_score(X_scaled, kmeans_model.predict(X_scaled))

    col1, col2 = st.columns(2)
    col1.metric("Silhouette Score", f"{sil:.3f}")
    col2.metric("Davies-Bouldin Index", f"{dbi:.3f}")

# ===============================
# MENU : PREDIKSI
# ===============================
elif menu == "Prediksi Cluster":
    st.subheader("🔍 Prediksi Cluster Data Baru")

    jenis_input = st.selectbox(
        "Pilih Jenis Bencana",
        JENIS_BENCANA
    )

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    input_data = []

    for col in numeric_cols:
        input_data.append(
            st.number_input(
                f"Masukkan nilai {col}",
                value=float(df[col].mean())
            )
        )

    if st.button("Prediksi Cluster"):
        new_df = pd.DataFrame([input_data], columns=numeric_cols)
        new_scaled = scaler.transform(new_df)
        cluster = kmeans_model.predict(new_scaled)[0]

        st.success(
            f"""
            **Jenis Bencana:** {jenis_input}  
            **Hasil Cluster:** Cluster {cluster}
            """
        )
