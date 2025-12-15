import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Projekt Jakosc 4.0 - AI w Produkcji",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- STYLE CSS ---
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
# --- SIDEBAR ---
with st.sidebar:
    st.title("O Projekcie")
    st.markdown("""
    **Temat:** Optymalizacja procesow w piekarni przemyslowej z wykorzystaniem AI.
    
    **Metodyka:**
    1. Pobranie danych z sensorow (IoT).
    2. **Safety Gate**: Odrzucenie odpadow krytycznych.
    3. **AI Classification**: Drzewo decyzyjne dla produktu bezpiecznego.
    4. **Dashboard**: Wizualizacja i system wsparcia decyzji.
    """)
    st.info("Dane pochodza z rzeczywistego procesu wypieku.")
# --- FUNKCJE ---
@st.cache_data
def load_data_full():
    try:
        # Wymuszenie silnika openpyxl
        df = pd.read_excel('data.xlsx', sheet_name='4 - klasyfikacja', header=2, engine='openpyxl')
        
        # Poprawa nazw kolumn
        cols = list(df.columns)
        if len(cols) >= 14:
            df.rename(columns={cols[12]: 'Status Bezpieczenstwa', cols[13]: 'Klasa jakosci'}, inplace=True)
        else:
            for c in cols:
                # Szukanie po fragmencie nazwy
                c_str = str(c).lower()
                if "odpad" in c_str:
                    df.rename(columns={c: 'Status Bezpieczenstwa'}, inplace=True)
                elif "premium" in c_str:
                    df.rename(columns={c: 'Klasa jakosci'}, inplace=True)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)
@st.cache_resource
def train_model(df_clean):
    try:
        X = df_clean[['Temperatura [°C]', 'Wilgotnosc [%]', 'Czas pieczenia [min]']] 
        y = df_clean['Klasa jakosci'].map({'PREMIUM': 1, 'STANDARD': 0})
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, class_weight='balanced', random_state=42)
        clf.fit(X, y)
        acc = accuracy_score(y, clf.predict(X))
        return clf, acc, X.columns
    except Exception as e:
        return None, 0, []
# --- GLOWNA LOGIKA ---
# 1. Wczytanie
df_raw, error_msg = load_data_full()
if df_raw.empty:
    st.error("Blad wczytywania pliku data.xlsx!")
    if error_msg:
        st.code(error_msg)
    st.stop()
# 2. Standaryzacja nazw kolumn (usuwamy polskie znaki z kluczy dla bezpieczenstwa kodu)
df_raw.rename(columns={
    'Temperatura [°C]': 'Temperatura [°C]', # bez zmian
    'Wilgotność [%]': 'Wilgotnosc [%]',
    'Status Bezpieczeństwa': 'Status Bezpieczenstwa',
    'Status Bezpieczenstwa': 'Status Bezpieczenstwa', # na wypadek roznych wersji
    'Klasa jakości': 'Klasa jakosci',
    'Klasa jakosci': 'Klasa jakosci'
}, inplace=True)
# 3. Filtr Safety Gate
n_total_raw = len(df_raw)
if 'Status Bezpieczenstwa' not in df_raw.columns:
    st.error("Nie znaleziono kolumny 'Status Bezpieczenstwa' (sprawdz naglowki w Excelu).")
    st.write("Dostepne kolumny:", list(df_raw.columns))
    st.stop()
df_safe = df_raw[df_raw['Status Bezpieczenstwa'] == 'OK'].copy()
n_safe = len(df_safe)
n_waste = n_total_raw - n_safe
# 4. Przygotowanie modelu
df_model = df_safe[df_safe['Klasa jakosci'].isin(['PREMIUM', 'STANDARD'])].copy()
# Konwersja liczb
cols_num = ['Temperatura [°C]', 'Wilgotnosc [%]', 'Czas pieczenia [min]', 'Zużycie energii [kWh]']
for c in cols_num:
    if c in df_model.columns:
        df_model[c] = pd.to_numeric(df_model[c], errors='coerce')
df_model = df_model.dropna(subset=['Temperatura [°C]', 'Klasa jakosci'])
model, acc, feature_names = train_model(df_model)
# --- INTERFEJS (UI) ---
st.markdown('<div class="main-header">QUALITY 4.0: SYSTEM EKSPERCKI</div>', unsafe_allow_html=True)
# Sekcja 0: Safety Gate
st.markdown("### 0. Bramka Bezpieczenstwa (Safety Gate)")
col_gate_desc, col_gate_chart = st.columns([2, 1])
with col_gate_desc:
    st.info("Produkty niespelniajace norm bezpieczenstwa (Odpad Krytyczny) sa odrzucane przed analiza jakosci.")
with col_gate_chart:
    st.metric("Odrzucone (Ryzyko)", f"{n_waste}")
    st.metric("Przekazane do Analizy", f"{n_safe}")
st.divider()
# Sekcja 1: KPI
st.header("1. Analiza Jakosci (Produkt Bezpieczny)")
if not df_model.empty:
    col1, col2 = st.columns(2)
    n_prem = len(df_model[df_model['Klasa jakosci'] == 'PREMIUM'])
    with col1:
        st.markdown(f"""<div class="kpi-card"><h3>Partie Handlowe</h3><h2>{len(df_model)}</h2></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-card"><h3>Udzial Premium</h3><h2>{(n_prem/len(df_model))*100:.1f}%</h2></div>""", unsafe_allow_html=True)
    st.write("")
    
    # Wykres
    tab1, tab2 = st.tabs(["Mapa Procesu", "Drzewo Decyzyjne"])
    with tab1:
        st.caption("Zielone punkty = Klasa PREMIUM")
        fig = px.scatter(
            df_model, 
            x='Temperatura [°C]', 
            y='Wilgotnosc [%]', 
            color='Klasa jakosci',
            color_discrete_map={'PREMIUM': '#00CC96', 'STANDARD': '#EF553B'},
            opacity=0.8
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        if model:
            fig_tree, ax = plt.subplots(figsize=(12, 6))
            fig_tree.patch.set_alpha(0)
            plot_tree(model, feature_names=list(feature_names), class_names=['STANDARD', 'PREMIUM'], filled=True, ax=ax)
            st.pyplot(fig_tree)
# Sekcja 2: Symulator
st.divider()
st.header("2. Symulator Procesu")
col_sim, col_res = st.columns([1, 1])
with col_sim:
    t_val = st.slider("Temperatura [C]", 150.0, 210.0, 178.0)
    h_val = st.slider("Wilgotnosc [%]", 10.0, 60.0, 35.0)
    time_val = st.slider("Czas [min]", 30.0, 70.0, 48.0)
with col_res:
    if t_val < 160 or t_val > 200:
        st.error("ODPAD KRYTYCZNY: Temperatura poza norma bezpieczenstwa!")
    else:
        if model:
            # Predykcja
            in_data = pd.DataFrame([[t_val, h_val, time_val]], columns=feature_names)
            pred = model.predict(in_data)[0]
            if pred == 1:
                st.markdown(f"""<div class="success-box"><h1>PREMIUM</h1></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="warning-box"><h1>STANDARD</h1></div>""", unsafe_allow_html=True)
                if not(169.96 < t_val <= 184.99):
                    st.warning(f"Temperatura {t_val}C poza idealnym zakresem (170-185C)")
                if not(30.02 < h_val <= 39.96):
                    st.warning(f"Wilgotnosc {h_val}% poza idealnym zakresem (30-40%)")
                if time_val > 49.95:
                    st.warning(f"Czas {time_val} min zbyt dlugi")
st.markdown("---")
st.caption("System v1.2 Final")
