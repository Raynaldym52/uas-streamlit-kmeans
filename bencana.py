import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ===============================
# FONT CONFIG
# ===============================
plt.rcParams["font.family"] = "DejaVu Sans"

# ===============================
# PAGE CONFIG
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
# LOAD DATA & MODEL
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("BencanaPWK2022.csv")

@st.cache_resource
def load_model():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")  # <<< WAJIB ADA
    return model, scaler, features

df_full = load_data()
kmeans_model, scaler, FEATURES = load_model()

# ===============================
# SIDEBAR
# ===============================
menu = st.sidebar.radio(
    "📌 Menu",
    ["Data", "Visualisasi Cluster", "Evaluasi Model", "Prediksi Cluster"]
)

selected_bencana = st.sidebar.multiselect(
    "Filter Jenis Bencana",
    JENIS_BENCANA,
    default=JENIS_BENCANA
)

# ===============================
# FILTER DATA (AMAN)
# ===============================
if 'jenis_bencana' in df_full.columns:
    df = df_full[df_full['jenis_bencana'].isin(selected_bencana)]
else:
    df = df_full.copy()

if len(df) < 2:
    st.warning("⚠️ Data terlalu sedikit untuk clustering.")
    st.stop()

# ===============================
# AMBIL FITUR SESUAI TRAINING
# ===============================
X = df[FEATURES]
X_scaled = scaler.transform(X)

# ===============================
# MENU DATA
# ===============================
if menu == "Data":
    st.subheader("📊 Dataset")
    st.dataframe(df, use_container_width=True)

# ===============================
# MENU VISUALISASI
# ===============================
elif menu == "Visualisasi Cluster":
    st.subheader("📉 Visualisasi PCA")

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
    st.pyplot(fig)

# ===============================
# MENU EVALUASI
# ===============================
elif menu == "Evaluasi Model":
    st.subheader("📊 Evaluasi K-Means")

    labels = kmeans_model.predict(X_scaled)

    if len(set(labels)) > 1:
        sil = silhouette_score(X_scaled, labels)
        dbi = davies_bouldin_score(X_scaled, labels)

        st.metric("Silhouette Score", f"{sil:.3f}")
        st.metric("Davies-Bouldin Index", f"{dbi:.3f}")
    else:
        st.warning("Silhouette Score tidak dapat dihitung (1 cluster).")

# ===============================
# MENU PREDIKSI
# ===============================
elif menu == "Prediksi Cluster":
    st.subheader("🔍 Prediksi Cluster")

    jenis = st.selectbox("Jenis Bencana", JENIS_BENCANA)

    input_data = {}
    for col in FEATURES:
        input_data[col] = st.number_input(
            f"Masukkan nilai {col}",
            value=float(df[col].mean())
        )

    if st.button("Prediksi"):
        new_df = pd.DataFrame([input_data])[FEATURES]
        new_scaled = scaler.transform(new_df)
        cluster = kmeans_model.predict(new_scaled)[0]

        st.success(
            f"""
            **Jenis Bencana:** {jenis}  
            **Hasil Cluster:** Cluster {cluster}
            """
        )
