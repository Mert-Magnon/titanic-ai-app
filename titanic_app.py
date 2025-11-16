import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 🎨 SAYFA AYARLARI
# ----------------------------------------------------
st.set_page_config(
    page_title="Titanic AI Predictor",
    page_icon="🚢",
    layout="centered"
)

# ----------------------------------------------------
# 🧭 NAVBAR
# ----------------------------------------------------
st.markdown("""
<style>
.navbar {
    background-color: #1f1f1f;
    padding: 14px;
    border-radius: 10px;
    color: white;
    font-size: 22px;
    text-align: center;
    margin-bottom: 20px;
}
</style>
<div class="navbar">🚢 Titanic Survival Predictor – AI Model</div>
""", unsafe_allow_html=True)

st.write("### Bu uygulama, Titanic yolcularının hayatta kalma olasılığını tahmin eder.")

# ----------------------------------------------------
# 🔄 MODEL YÜKLEME
# ----------------------------------------------------
with st.spinner("🔄 Model yükleniyor... Lütfen bekleyin"):
    model = joblib.load("titanic_model.pkl")

# Kullanılacak özellikler
feature_names = ["Pclass", "Sex", "Age", "Fare", "Embarked"]

# ----------------------------------------------------
# 🧮 KULLANICI GİRDİLERİ
# ----------------------------------------------------
st.sidebar.header("Yolcu Bilgileri")

pclass = st.sidebar.selectbox("Bilet Sınıfı", [1, 2, 3])
sex = st.sidebar.selectbox("Cinsiyet", ["male", "female"])
age = st.sidebar.slider("Yaş", 1, 80, 28)
fare = st.sidebar.slider("Bilet Ücreti", 0, 500, 50)
embarked = st.sidebar.selectbox("Biniş Limanı", ["S", "C", "Q"])

# Kategorik dönüşümler
sex_map = {"male": 0, "female": 1}
embark_map = {"S": 0, "C": 1, "Q": 2}

input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex_map[sex],
    "Age": age,
    "Fare": fare,
    "Embarked": embark_map[embarked]
}])

# ----------------------------------------------------
# 📋 GİRDİ ÖZETİ
# ----------------------------------------------------
st.subheader("📋 Girdi Özeti")
st.table(input_data)

# ----------------------------------------------------
# 🤖 TAHMİN
# ----------------------------------------------------
if st.button("🚀 Tahmin Et"):
    prob = model.predict_proba(input_data)[0][1]
    yuzde = prob * 100

    # Sonuç kutusu
    st.success(f"### 💡 Hayatta Kalma Olasılığı: **%{yuzde:.2f}**")

    # Metric kartı
    st.metric("Olasılık (%)", f"%{yuzde:.2f}")

    # İlerleme çubuğu
    st.progress(prob)

    # ------------------------------------------------
    # 📊 Feature Importance Grafiği
    # ------------------------------------------------
    st.subheader("📈 Model Özellik Önem Grafiği")

    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(feature_names, importances)
    ax.set_xlabel("Önem Derecesi")
    ax.set_title("Feature Importance")
    st.pyplot(fig)
