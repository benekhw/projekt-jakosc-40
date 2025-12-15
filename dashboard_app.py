import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# --- KONFIGURACJA STRONY (PREMIUM) ---
st.set_page_config(
    page_title="Quality 4.0 Pro | AI Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# --- STYLE CSS (MODERN UI) ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; color: #1F2937; }
    .hero-title { font-size: 3.5rem; font-weight: 800; background: -webkit-linear-gradient(45deg, #2563EB, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0.5rem; }
    .hero-subtitle { font-size: 1.2rem; color: #6B7280; text-align: center; margin-bottom: 3rem; }
    .kpi-container { background: linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%); padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #E5E7EB; text-align: center; }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #111827; }
    .kpi-label { font-size: 0.9rem; color: #6B7280; text-transform: uppercase; letter-spacing: 1px; }
    .safety-gate { padding: 20px; background-color: #FEF2F2; border-right: 5px solid #EF4444; border-radius: 8px; margin-bottom: 20px; }
    .safety-title { color: #B91C1C; font-weight: bold; font-size: 1.1rem; }
    .premium-gate { padding: 20px; background-color: #ECFDF5; border-right: 5px solid #10B981; border-radius: 8px; margin-bottom: 20px; }
    .premium-title { color: #047857; font-weight: bold; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)
# --- FUNKCJE DANYCH (PRZYWROCONE Z DZIALAJACEJ WERSJI) ---
@st.cache_data
def load_and_fix_data():
    # 1. Wczytanie (próbujemy CSV, fallback do Excela)
    try:
        # header=2 -> wiersz 3 to nagłówki
        df = pd.read_csv('data.csv', header=2, sep=None, engine='python')
    except:
        try:
            df = pd.read_excel('data.xlsx', sheet_name='4 - klasyfikacja', header=2, engine='openpyxl')
        except:
            return pd.DataFrame() # Pusty jak nic nie działa
    if df.empty: return df
    # 2. NAPRAWA NAZW KOLUMN (NA SZTYWNO PO INDEKSACH)
    cols = list(df.columns)
    
    # Mapa nowych nazw
    rename_map = {
        cols[6]: 'Temperatura',   # "Temperatura [°C]"
        cols[7]: 'Wilgotnosc',    # "Wilgotność [%]"
        cols[9]: 'Czas',          # "Czas pieczenia [min]"
    }
    
    # Tylko jeśli mamy wystarczająco kolumn
    if len(cols) >= 14:
        rename_map[cols[12]] = 'Status'  # Ta długa nazwa "Jeśli temperatura..."
        rename_map[cols[13]] = 'Jakosc'  # Ta długa nazwa "Jeśli produkt..."
    
    df = df.rename(columns=rename_map)
    
    # Konwersja numeryczna (na wszelki wypadek)
    for c in ['Temperatura', 'Wilgotnosc', 'Czas']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df
@st.cache_resource
def train_model(df):
    try:
        X = df[['Temperatura', 'Wilgotnosc', 'Czas']]
        y = df['Jakosc'].map({'PREMIUM': 1, 'STANDARD': 0})
        clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
        clf.fit(X, y)
        acc = accuracy_score(y, clf.predict(X))
        return clf, acc
    except:
        return None, 0
# --- GLOWNA LOGIKA ---
df_raw = load_and_fix_data()
if df_raw.empty:
    st.error("⚠️ Brak danych. Upewnij się, że plik data.csv/xlsx jest poprawny na GitHubie.")
    st.stop()
# Filtrowanie i logika Safety Gate
df_safe = df_raw[df_raw['Status'] == 'OK'].copy()
df_waste = df_raw[df_raw['Status'] != 'OK'].copy()
n_waste = len(df_waste)
n_safe = len(df_safe)
# Model Data (tylko poprawne klasy i pełne dane)
df_model = df_safe[df_safe['Jakosc'].isin(['PREMIUM', 'STANDARD'])].dropna(subset=['Temperatura', 'Wilgotnosc', 'Czas', 'Jakosc'])
model, accuracy = train_model(df_model)
# Liczymy metryki
n_premium = len(df_model[df_model['Jakosc'] == 'PREMIUM'])
ratio_premium = n_premium / len(df_model) * 100 if len(df_model) > 0 else 0
# --- INTERFEJS (PREMIUM LOOK) ---
st.markdown('<div class="hero-title">QUALITY 4.0</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">System Wspierania Decyzji i Optymalizacji Procesu Wypieku</div>', unsafe_allow_html=True)
# TABS
tab_overview, tab_analysis, tab_sim, tab_about = st.tabs([
    "📊 Panel Zarządczy (KPI)", 
    "🔍 Analiza AI & Wzorce", 
    "🎛️ Digital Twin (Symulator)", 
    "📝 Dokumentacja"
])
# TAB 1: OVERVIEW
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"""<div class="kpi-container"><div class="kpi-value">{len(df_raw)}</div><div class="kpi-label">Wszystkie Partie</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="kpi-container"><div class="kpi-value" style="color:#EF4444">{n_waste}</div><div class="kpi-label">Odrzuty (Safety)</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="kpi-container"><div class="kpi-value" style="color:#10B981">{ratio_premium:.1f}%</div><div class="kpi-label">Wskaźnik Premium</div></div>""", unsafe_allow_html=True)
    with c4: st.markdown(f"""<div class="kpi-container"><div class="kpi-value">{df_model['Temperatura'].mean():.1f}°C</div><div class="kpi-label">Śr. Temperatura</div></div>""", unsafe_allow_html=True)
    st.write("")
    st.markdown("### 🚦 Przepływ Procesu")
    col_flow1, col_flow2, col_flow3 = st.columns(3)
    with col_flow1: st.markdown("""<div class="safety-gate"><div class="safety-title">1. INPUT</div>Monitoring IoT (Ciągły)</div>""", unsafe_allow_html=True)
    with col_flow2: st.markdown(f"""<div class="safety-gate" style="border-color:#F59E0B;background:#FFFBEB"><div class="safety-title" style="color:#D97706">2. SAFETY GATE</div>Odrzucono: <b>{n_waste}</b> partii</div>""", unsafe_allow_html=True)
    with col_flow3: st.markdown(f"""<div class="premium-gate"><div class="premium-title">3. AI CLASSIFICATION</div>Premium: {n_premium} / Standard: {len(df_model)-n_premium}</div>""", unsafe_allow_html=True)
# TAB 2: AI ANALIZY
with tab_analysis:
    st.markdown("### 🧠 Odkrywanie Wzorców")
    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        st.info("Złota Strefa (Zielona ramka) to obszar, w którym produkujemy klasę PREMIUM.")
        if not df_model.empty:
            # Używamy prostego scattera, który na pewno działa
            fig = px.scatter(df_model, x='Temperatura', y='Wilgotnosc', color='Jakosc',
                             color_discrete_map={'PREMIUM': '#10B981', 'STANDARD': '#EF4444'},
                             title="Mapa Jakości", opacity=0.7)
            # Dodajemy złotą ramkę
            fig.add_shape(type="rect", x0=170, y0=30, x1=185, y1=40, line=dict(color="green", width=2))
            st.plotly_chart(fig, use_container_width=True)
    with col_a2:
        st.markdown("**Reguły Sukcesu:**")
        st.markdown("1. **Temp:** 170-185°C\n2. **Wilgotność:** 30-40%\n3. **Czas:** < 50 min")
        st.metric("Dokładność Modelu", f"{accuracy*100:.1f}%")
# TAB 3: SYMULATOR
with tab_sim:
    st.markdown("### 🎛️ Digital Twin")
    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        s_t = st.slider("Temperatura", 150, 210, 178)
        s_h = st.slider("Wilgotność", 10, 60, 35)
        s_c = st.slider("Czas", 30, 70, 48)
    with c_s2:
        if s_t < 160 or s_t > 200:
            st.error("⛔ ODPAD KRYTYCZNY (Temp)")
            safe = False
        else:
            st.success("✅ Parametry OK")
            safe = True
    with c_s3:
        if safe and model:
            res = model.predict([[s_t, s_h, s_c]])[0]
            if res == 1:
                st.balloons()
                st.success("💎 PREMIUM")
            else:
                st.warning("⚠️ STANDARD")
# TAB 4: DOCS
with tab_about:
    st.markdown("### 📄 Dokumentacja")
    st.write("Celem projektu jest detekcja jakości przy użyciu AI. (Pełny opis w kodzie...)")
    st.caption("© 2024 Quality 4.0")
