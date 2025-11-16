import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ========== UYGULAMA AYARLARI ==========
st.set_page_config(page_title="🚢 Titanic AI Tahmin Uygulaması", page_icon="🚢", layout="wide")

# Model yükle
model = joblib.load("titanic_model.pkl")

# ========== SIDEBAR ==========
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/320px-RMS_Titanic_3.jpg", width=220)
st.sidebar.title("🧭 Menü")

page = st.sidebar.radio("Sayfa Seç:", ["🎯 Tahmin Aracı", "📊 Model Bilgisi", "ℹ️ Hakkında"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Sürümü:** v1.0")
st.sidebar.markdown("**Eğitim Seti:** Titanic (Kaggle)")
st.sidebar.markdown("**Yapay Zeka:** Random Forest Classifier")

# ========== 1️⃣ TAHMİN ARACI ==========
if page == "🎯 Tahmin Aracı":
    st.title("🚢 Titanic Hayatta Kalma Tahmin Aracı")
    st.markdown("Yolcu bilgilerini gir ve modelin tahminini **olasılıkla birlikte** gör 👇")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Bilet Sınıfı (Pclass)", [1, 2, 3])
        sex = st.selectbox("Cinsiyet", ["male", "female"])
        age = st.slider("Yaş", 0, 80, 25)

    with col2:
        sibsp = st.number_input("Kardeş/Eş Sayısı (SibSp)", 0, 8, 0)
        parch = st.number_input("Ebeveyn/Çocuk Sayısı (Parch)", 0, 6, 0)
        fare = st.slider("Bilet Ücreti (Fare)", 0, 500, 50)
        embarked = st.selectbox("Biniş Limanı (Embarked)", ["C", "Q", "S"])

    # Özellik mühendisliği
    sex = 1 if sex == "male" else 0
    family_size = sibsp + parch + 1
    embarked_mapping = {"C": 0, "Q": 1, "S": 2}
    embarked = embarked_mapping[embarked]

    data = pd.DataFrame([[pclass, sex, age, fare, embarked, family_size]],
                        columns=["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"])

    if st.button("🚀 Tahmini Gör"):
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1] * 100

        st.markdown("---")
        col_pred, col_gauge = st.columns([1, 2])

        with col_pred:
            if prediction == 1:
                st.success("💚 Bu yolcu **büyük ihtimalle hayatta kalırdı!**")
            else:
                st.error("💀 Bu yolcu **muhtemelen hayatta kalamazdı.**")
            st.markdown(f"### 🔢 Olasılık: **{probability:.2f}%**")

        # Gauge grafiği
        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                title={'text': "Hayatta Kalma Olasılığı", 'font': {'size': 22}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green" if prediction == 1 else "red"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffcccc"},
                        {'range': [50, 75], 'color': "#fff2cc"},
                        {'range': [75, 100], 'color': "#ccffcc"},
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': probability}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

# ========== 2️⃣ MODEL BİLGİSİ ==========
elif page == "📊 Model Bilgisi":
    st.title("📊 Model Bilgisi ve Özellik Önemi")
    st.markdown("Bu model, **Random Forest Classifier** algoritması kullanılarak eğitilmiştir.")

    importances = model.feature_importances_
    feature_names = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]

    imp_df = pd.DataFrame({
        "Özellik": feature_names,
        "Önem": importances
    }).sort_values("Önem", ascending=True)

    fig_imp = px.bar(
        imp_df,
        x="Önem",
        y="Özellik",
        orientation="h",
        color="Önem",
        color_continuous_scale="Viridis",
        title="Modelin Özellik Önemi Dağılımı",
        height=450
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info("🔍 Grafik, modelin tahminlerinde hangi değişkenlere daha fazla ağırlık verdiğini gösterir.")

# ========== 3️⃣ HAKKINDA ==========
elif page == "ℹ️ Hakkında":
    st.title("ℹ️ Uygulama Hakkında")
    st.markdown("""
    Bu proje, **Yapay Zeka ve Makine Öğrenimi Öğrenme Yolculuğu** kapsamında geliştirilmiştir.  
    Titanic veri seti, makine öğrenmesi algoritmalarını denemek için en popüler veri setlerinden biridir.  
    Bu uygulama, `RandomForestClassifier` modeliyle bir yolcunun hayatta kalma olasılığını tahmin eder.

    **Kullanılan Teknolojiler:**
    - Python 3.11+
    - Streamlit
    - Pandas
    - Scikit-learn
    - Plotly

    **Hazırlayan:** 👨‍💻 [Yusuf Mert Özaydın]  
    **Amaç:** Makine öğrenimi temellerini, model analizi ve kullanıcı arayüzü tasarımıyla birleştirmek.
    """)
    st.markdown("---")
    st.caption("© 2025 Yapay Zeka Öğrenme Serisi | Eğitim Amaçlı Kullanım")
