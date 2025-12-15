import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
# --- FUNKCJE DANYCH ---
@st.cache_data
def load_and_process_data():
    # 1. Wczytanie
    try:
        df = pd.read_csv('data.csv', header=2, sep=None, engine='python')
    except:
        try:
            df = pd.read_excel('data.xlsx', sheet_name='4 - klasyfikacja', header=2, engine='openpyxl')
        except:
            return pd.DataFrame()
    if df.empty: return df
    # 2. Mapowanie
    cols = list(df.columns)
    rename_map = {
        cols[6]: 'Temperatura',
        cols[7]: 'Wilgotnosc',
        cols[9]: 'Czas',
    }
    if len(cols) >= 14:
        rename_map[cols[12]] = 'Status'
        rename_map[cols[13]] = 'Jakosc'
    
    df = df.rename(columns=rename_map)
    
    # 3. Konwersja i CZYSZCZENIE (Kluczowe dla Plotly)
    for c in ['Temperatura', 'Wilgotnosc', 'Czas']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Usuwamy wiersze, gdzie kluczowe dane są NaN (to naprawia błąd Plotly)
    df = df.dropna(subset=['Temperatura', 'Wilgotnosc', 'Czas', 'Status', 'Jakosc'])
    
    return df
@st.cache_resource
def train_ai_model(df):
    try:
        X = df[['Temperatura', 'Wilgotnosc', 'Czas']]
        y = df['Jakosc'].map({'PREMIUM': 1, 'STANDARD': 0})
        clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
        clf.fit(X, y)
        acc = accuracy_score(y, clf.predict(X))
        return clf, acc
    except:
        return None, 0
# --- START ---
df_raw = load_and_process_data()
if df_raw.empty:
    st.error("⚠️ Brak danych. Upewnij się, że plik data.csv/xlsx jest poprawny.")
    st.stop()
# --- PROCES ---
df_safe = df_raw[df_raw['Status'] == 'OK'].copy()
df_waste = df_raw[df_raw['Status'] != 'OK'].copy()
n_waste = len(df_waste)
n_safe = len(df_safe)
df_model = df_safe[df_safe['Jakosc'].isin(['PREMIUM', 'STANDARD'])].copy()
model, accuracy = train_ai_model(df_model)
if model is None:
    st.warning("⚠️ Zbyt mało danych do wytrenowania modelu.")
else:
    n_premium = len(df_model[df_model['Jakosc'] == 'PREMIUM'])
    ratio_premium = n_premium / len(df_model) * 100 if len(df_model) > 0 else 0
# --- UI ---
st.markdown('<div class="hero-title">QUALITY 4.0</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">System Wspierania Decyzji i Optymalizacji Procesu Wypieku</div>', unsafe_allow_html=True)
tab_overview, tab_analysis, tab_sim, tab_about = st.tabs([
    "📊 Panel Zarządczy (KPI)", "🔍 Analiza AI & Wzorce", "🎛️ Digital Twin (Symulator)", "📝 Dokumentacja"
])
# TAB 1: KPI
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
    with col_flow2: st.markdown(f"""<div class="safety-gate" style="border-color:#F59E0B;background:#FFFBEB"><div class="safety-title" style="color:#D97706">2. SAFETY GATE</div>Odrzucono: <b>{n_waste}</b></div>""", unsafe_allow_html=True)
    with col_flow3: st.markdown(f"""<div class="premium-gate"><div class="premium-title">3. AI CLASSIFICATION</div>Premium: {n_premium}</div>""", unsafe_allow_html=True)
# TAB 2: AI
with tab_analysis:
    st.markdown("### 🧠 Odkrywanie Wiedzy")
    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        st.info("Zielona strefa to optymalne parametry produkcji.")
        if not df_model.empty:
            # Usunięcie template="plotly_white" dla pewności i obsługa size
            fig_scatter = px.scatter(
                df_model, x='Temperatura', y='Wilgotnosc', color='Jakosc',
                size='Czas', # Size wymaga number > 0. dropna() wyzej to zapewnia, ale...
                color_discrete_map={'PREMIUM': '#10B981', 'STANDARD': '#EF4444'},
                title="Mapa Jakości: Temperatura vs Wilgotność (Wielkość = Czas)",
                hover_data=['Czas']
            )
            # Ręczne ustawienie template w update_layout jest bezpieczniejsze
            fig_scatter.update_layout(template="simple_white")
            
            # Złota strefa
            fig_scatter.add_shape(type="rect", x0=170, y0=30, x1=185, y1=40, line=dict(color="#10B981", width=2))
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col_a2:
        st.markdown("**Reguły:**")
        st.markdown("1. **Temp:** 170-185°C (Optimum)\n2. **Wilgotność:** 30-40%\n3. **Czas:** < 50 min")
        st.metric("Dokładność AI", f"{accuracy*100:.1f}%")
# TAB 3: SYMULATOR
with tab_sim:
    st.markdown("### 🎛️ Digital Twin")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.subheader("1. Ustawienia")
        s_temp = st.slider("Temp [°C]", 150, 210, 178)
        s_hum = st.slider("Wilgotność [%]", 10, 60, 35)
        s_time = st.slider("Czas [min]", 30, 70, 48)
    with col_s2:
        st.subheader("2. Safety Gate")
        safe = True
        if s_temp < 160 or s_temp > 200:
            st.error(f"⛔ ODPAD KRYTYCZNY (Temp {s_temp}°C)")
            safe = False
        else:
            st.success("✅ Parametry Bezpieczne")
    with col_s3:
        st.subheader("3. Wynik AI")
        if safe and model:
            res = model.predict([[s_temp, s_hum, s_time]])[0]
            if res == 1:
                st.balloons()
                st.success("💎 PREMIUM")
            else:
                st.warning("⚠️ STANDARD")
# TAB 4: DOCS
with tab_about:
    st.markdown("### 📄 Dokumentacja")
    st.markdown("System analizuje parametry procesu wypieku w celu klasyfikacji jakości (Premium/Standard) i odrzucania odpadów krytycznych.")
    st.caption("© 2024 Quality 4.0 Pro")
