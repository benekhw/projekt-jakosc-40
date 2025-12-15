import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Projekt Jakość 4.0 - AI w Produkcji",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- ŁADOWANIE JĘZYKA CSS (Fix for Dark Mode) ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        background: -webkit-linear-gradient(45deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    .kpi-card {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        text-align: center;
    }
    .success-box {
        padding: 20px;
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 5px solid #10B981;
        color: #10B981; 
        border-radius: 5px;
    }
    .warning-box {
        padding: 20px;
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 5px solid #F59E0B;
        color: #F59E0B;
        border-radius: 5px;
    }
    .info-box {
        padding: 15px;
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 5px solid #3B82F6;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)
# --- SIDEBAR: O PROJEKCIE (Kontekst) ---
with st.sidebar:
    st.title("O Projekcie")
    st.markdown("""
    **Temat:** Optymalizacja procesów w piekarni przemysłowej z wykorzystaniem AI.
    
    **Autor:** Zespół Projektowy
    
    **Metodyka:**
    1. Pobranie danych z sensorów (IoT).
    2. **Safety Gate**: Odrzucenie odpadów krytycznych.
    3. **AI Classification**: Drzewo decyzyjne dla produktu bezpiecznego.
    4. **Dashboard**: Wizualizacja i system wsparcia decyzji.
    """)
    st.info("Dane pochodzą z rzeczywistego procesu wypieku bagietek na 3 liniach produkcyjnych.")
# --- FUNKCJE DANYCH ---
@st.cache_data
def load_data_full():
    try:
        df = pd.read_excel('data.xlsx', sheet_name='4 - klasyfikacja', header=2)
        
        # Zmiana nazw cols
        cols = list(df.columns)
        if len(cols) >= 14:
            df.rename(columns={cols[12]: 'Status Bezpieczeństwa', cols[13]: 'Klasa jakości'}, inplace=True)
        else:
            for c in cols:
                if "odpad" in str(c).lower():
                    df.rename(columns={c: 'Status Bezpieczeństwa'}, inplace=True)
                elif "premium" in str(c).lower():
                    df.rename(columns={c: 'Klasa jakości'}, inplace=True)
        return df
    except:
        return pd.DataFrame()
@st.cache_resource
def train_model(df_clean):
    try:
        X = df_clean[['Temperatura [°C]', 'Wilgotność [%]', 'Czas pieczenia [min]']] 
        y = df_clean['Klasa jakości'].map({'PREMIUM': 1, 'STANDARD': 0})
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, class_weight='balanced', random_state=42)
        clf.fit(X, y)
        acc = accuracy_score(y, clf.predict(X))
        return clf, acc, X.columns
    except:
        return None, 0, []
# --- LOAD DATA ---
df_raw = load_data_full()
if df_raw.empty:
    st.error("Brak pliku data.xlsx! Wgraj plik Excel do repozytorium.")
    st.stop()
# LOGIKA FILTRA (Bramka Bezpieczeństwa)
n_total_raw = len(df_raw)
df_safe = df_raw[df_raw['Status Bezpieczeństwa'] == 'OK'].copy()
n_safe = len(df_safe)
n_waste = n_total_raw - n_safe
# Przygotowanie danych do modelu (tylko bezpieczne)
df_model = df_safe[df_safe['Klasa jakości'].isin(['PREMIUM', 'STANDARD'])].dropna(subset=['Temperatura [°C]', 'Klasa jakości'])
for c in ['Temperatura [°C]', 'Wilgotność [%]', 'Czas pieczenia [min]', 'Zużycie energii [kWh]']:
    df_model[c] = pd.to_numeric(df_model[c], errors='coerce')
df_model = df_model.dropna()
model, acc, feature_names = train_model(df_model)
# --- HERO ---
st.markdown('<div class="main-header">QUALITY 4.0: SYSTEM EKSPERCKI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automatyczna klasyfikacja jakości wypieku i detekcja anomalii procesowych</div>', unsafe_allow_html=True)
# --- SEKCJA MODELU BIZNESOWEGO (BRAMKA BEZPIECZEŃSTWA) ---
st.markdown("### 0. Bramka Bezpieczeństwa (Safety Gate)")
col_gate_desc, col_gate_chart = st.columns([2, 1])
with col_gate_desc:
    st.markdown("""
    <div class="info-box">
    <b>Dlaczego nie analizujemy wszystkiego?</b><br>
    W produkcji żywności bezpieczeństwo jest nadrzędne. Zanim ocenimy czy produkt jest PREMIUM czy STANDARD,
    musimy sprawdzić, czy nadaje się do spożycia.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Algorytm Postępowania:**
    1.  **ODPAD KRYTYCZNY**: Jeśli temperatura < 160°C lub > 200°C -> *Utylizacja natychmiastowa* (Zagrożenie mikrobiologiczne lub spalenie). Tych danych **NIE wprowadzamy** do modelu jakości, bo nie są to produkty handlowe.
    2.  **PRODUKT BEZPIECZNY**: Dopiero produkt spełniający normy bezpieczeństwa trafia do sekcji klasyfikacji (AI).
    """)
with col_gate_chart:
    st.metric("Odrzucone (Odpad Krytyczny)", f"{n_waste}", delta="- Ryzyko", delta_color="inverse")
    st.metric("Przekazane do Analizy Jakości", f"{n_safe}", delta="Bezpieczne")
st.divider()
# --- KPI (Dla bezpiecznych) ---
st.header("1. Analiza Jakości (Produkt Bezpieczny)")
col1, col2, col3, col4 = st.columns(4)
n_model = len(df_model)
n_prem = len(df_model[df_model['Klasa jakości'] == 'PREMIUM'])
with col1:
    st.markdown(f"""<div class="kpi-card"><h3>Partie Handlowe</h3><h2>{n_model}</h2><small>Po odrzuceniu odpadów</small></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="kpi-card"><h3>Udział Premium</h3><h2>{(n_prem/n_model)*100:.1f}%</h2><small>Cel: > 25%</small></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="kpi-card"><h3>Śr. Czas</h3><h2>{df_model['Czas pieczenia [min]'].mean():.1f} min</h2></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="kpi-card"><h3>Śr. Temperatura</h3><h2>{df_model['Temperatura [°C]'].mean():.1f} °C</h2></div>""", unsafe_allow_html=True)
st.write("")
# --- WIZUALIZACJA ---
tab_viz, tab_tree = st.tabs(["Mapa Procesu (Scatter Plot)", "Struktura Drzewa (Model)"])
with tab_viz:
    st.caption("Wykres pokazuje 'Złotą Strefę' parametrów dla klasy PREMIUM (Zielone punkty).")
    fig = px.scatter(
        df_model, 
        x='Temperatura [°C]', 
        y='Wilgotność [%]', 
        color='Klasa jakości',
        symbol='Klasa jakości',
        size='Czas pieczenia [min]',
        color_discrete_map={'PREMIUM': '#00CC96', 'STANDARD': '#EF553B'},
        hover_data=['Numer partii'],
        opacity=0.8
    )
    # Złota strefa
    fig.add_shape(type="rect",
        x0=170, y0=30, x1=185, y1=40,
        line=dict(color="Green", width=2),
        fillcolor="Green",
        opacity=0.1
    )
    fig.update_layout(xaxis_title="Temperatura Pieca [°C]", yaxis_title="Wilgotność [%]", height=500)
    st.plotly_chart(fig, use_container_width=True)
with tab_tree:
    col_t_desc, col_t_img = st.columns([1, 2])
    with col_t_desc:
        st.markdown("### Jak działa model?")
        st.markdown("""
        Drzewo decyzyjne dzieli proces na proste pytania (węzły). 
        
        **Zasady PREMIUM:**
        1.  **Czas**: Musi być krótki (< 50 min), aby zachować świeżość.
        2.  **Temperatura**: Musi być idealna (170-185°C). Zbyt niska = niedopiek, zbyt wysoka = przesuszenie.
        3.  **Wilgotność**: 30-40% zapewnia chrupkość.
        """)
    with col_t_img:
        if model:
            fig_tree, ax = plt.subplots(figsize=(12, 6))
            # Ustawienie kolorów tła pod temat Streamlit (przezroczyste)
            fig_tree.patch.set_alpha(0)
            plot_tree(model, feature_names=list(feature_names), class_names=['STANDARD', 'PREMIUM'], 
                     filled=True, rounded=True, fontsize=10, ax=ax)
            st.pyplot(fig_tree)
st.divider()
# --- SYMULATOR ---
st.header("2. Symulator Procesu (Digital Twin)")
st.caption("Zmień parametry suwakami, aby sprawdzić predykcję modelu.")
col_sim, col_res = st.columns([1, 1])
with col_sim:
    temp_val = st.slider("Temperatura [°C]", 150.0, 210.0, 178.0, step=0.1)
    hum_val = st.slider("Wilgotność [%]", 10.0, 60.0, 35.0, step=0.1)
    time_val = st.slider("Czas pieczenia [min]", 30.0, 70.0, 48.0, step=0.5)
with col_res:
    # 1. BRAMKA BEZPIECZEŃSTWA (Logika symulatora)
    is_safe = True
    safety_msg = ""
    
    if temp_val < 160 or temp_val > 200:
        is_safe = False
        safety_msg = "ODPAD KRYTYCZNY: Temperatura poza zakresem bezpieczeństwa (160-200°C)!"
    
    if not is_safe:
        st.error(safety_msg)
        st.markdown("Partia zostaje odrzucona automatycznie przed analizą jakości.")
    else:
        # 2. MODEL JAKOŚCI
        if model:
            input_df = pd.DataFrame([[temp_val, hum_val, time_val]], columns=feature_names)
            pred = model.predict(input_df)[0]
            
            if pred == 1:
                st.markdown(f"""<div class="success-box"><h1>PREMIUM</h1><p>Parametry w normie.</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="warning-box"><h1>STANDARD</h1></div>""", unsafe_allow_html=True)
                
                st.write("**Przyczyna obniżenia jakości:**")
                # Dokładna diagnostyka
                if not(169.96 < temp_val <= 184.99):
                    st.warning(f"Temperatura {temp_val}°C poza idealnym zakresem (170-185°C)")
                if not(30.02 < hum_val <= 39.96):
                    st.warning(f"Wilgotność {hum_val}% poza idealnym zakresem (30-40%)")
                if time_val > 49.95:
                    st.warning(f"Czas {time_val} min przekracza limit 50 min")
st.markdown("---")
st.caption("System v1.0 | Projekt Studencki | Analiza Danych Sp. z o.o.")
