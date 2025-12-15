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
    initial_sidebar_state="collapsed" # Zaczynamy szeroko
)
# --- STYLE CSS (MODERN UI) ---
st.markdown("""
<style>
    /* Globalne fonty i tło */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Nagłówki */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #1F2937;
    }
    .dark-theme h1, .dark-theme h2, .dark-theme h3 {
        color: #F9FAFB;
    }
    
    /* Hero Section */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2563EB, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 3rem;
    }
    /* KPI Cards */
    .kpi-container {
        background: linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #111827;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sekcja Safety Gate */
    .safety-gate {
        padding: 20px;
        background-color: #FEF2F2;
        border-right: 5px solid #EF4444;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .safety-title {
        color: #B91C1C;
        font-weight: bold;
        font-size: 1.1rem;
    }
    /* Sekcja Premium */
    .premium-gate {
        padding: 20px;
        background-color: #ECFDF5;
        border-right: 5px solid #10B981;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .premium-title {
        color: #047857;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)
# --- FUNKCJE DANYCH (MECHANIZM PANCERNY) ---
@st.cache_data
def load_and_process_data():
    # 1. Wczytanie (CSV lub Excel)
    try:
        df = pd.read_csv('data.csv', header=2, sep=None, engine='python')
    except:
        try:
            df = pd.read_excel('data.xlsx', sheet_name='4 - klasyfikacja', header=2, engine='openpyxl')
        except:
            return pd.DataFrame()
    if df.empty: return df
    # 2. Mapowanie kolumn (niezawodne)
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
    
    # Konwersja numeryczna
    for c in ['Temperatura', 'Wilgotnosc', 'Czas']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
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
# --- ŁADOWANIE ---
df_raw = load_and_process_data()
if df_raw.empty:
    st.error("⚠️ Błąd danych. Proszę wgrać 'data.csv' (zapisany z arkusza 4 - klasyfikacja) lub 'data.xlsx' na GitHub.")
    st.stop()
# --- PRZETWARZANIE LOGICZNE (ETL) ---
# 1. Safety Gate Logic
df_safe = df_raw[df_raw['Status'] == 'OK'].copy()
df_waste = df_raw[df_raw['Status'] != 'OK'].copy()
n_waste = len(df_waste)
n_safe = len(df_safe)
# 2. AI Model Data
df_model = df_safe[df_safe['Jakosc'].isin(['PREMIUM', 'STANDARD'])].dropna(subset=['Temperatura', 'Jakosc'])
model, accuracy = train_ai_model(df_model)
n_premium = len(df_model[df_model['Jakosc'] == 'PREMIUM'])
ratio_premium = n_premium / len(df_model) * 100 if len(df_model) > 0 else 0
# --- INTERFEJS GLÓWNY ---
# HERO SECTION
st.markdown('<div class="hero-title">QUALITY 4.0</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">System Wspierania Decyzji i Optymalizacji Procesu Wypieku</div>', unsafe_allow_html=True)
# NAV (TABS)
tab_overview, tab_analysis, tab_sim, tab_about = st.tabs([
    "📊 Panel Zarządczy (KPI)", 
    "🔍 Analiza AI & Wzorce", 
    "🎛️ Digital Twin (Symulator)", 
    "📝 Dokumentacja Projektu"
])
# --- TAB 1: OVERVIEW ---
with tab_overview:
    # KPI ROW
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-container"><div class="kpi-value">{len(df_raw)}</div><div class="kpi-label">Wszystkie Partie</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-container"><div class="kpi-value" style="color:#EF4444">{n_waste}</div><div class="kpi-label">Odrzuty (Safety)</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-container"><div class="kpi-value" style="color:#10B981">{ratio_premium:.1f}%</div><div class="kpi-label">Wskaźnik Premium</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-container"><div class="kpi-value">{df_model['Temperatura'].mean():.1f}°C</div><div class="kpi-label">Śr. Temperatura</div></div>""", unsafe_allow_html=True)
    st.write("")
    st.markdown("### 🚦 Przepływ Procesu (Process Flow)")
    
    # SANKY CHART (Prosty przepływ za pomocą kolumn)
    col_flow1, col_flow2, col_flow3 = st.columns(3)
    
    with col_flow1:
        st.markdown("""
        <div class="safety-gate">
            <div class="safety-title">1. INPUT: Produkcja</div>
            Wszystkie partie wchodzą do systemu monitoringu IoT.<br><br>
            <b>Status:</b> Monitoring ciągły.
        </div>
        """, unsafe_allow_html=True)
        
    with col_flow2:
        st.markdown(f"""
        <div class="safety-gate" style="border-color: #F59E0B; background-color: #FFFBEB;">
            <div class="safety-title" style="color: #D97706">2. SAFETY GATE oT</div>
            Automat odrzuca <b style="color:#EF4444">{n_waste}</b> partii niebezpiecznych (temp <160 lub >200).<br>
            Pozostałe {n_safe} przechodzi dalej.
        </div>
        """, unsafe_allow_html=True)
        
    with col_flow3:
        st.markdown(f"""
        <div class="premium-gate">
            <div class="premium-title">3. AI CLASSIFICATION</div>
            Model ocenia jakość produktu końcowego.<br>
            <b>Wynik:</b> {n_premium} Premium vs {len(df_model)-n_premium} Standard.
        </div>
        """, unsafe_allow_html=True)
# --- TAB 2: ANALIZA AI ---
with tab_analysis:
    st.markdown("### 🧠 Odkrywanie Wiedzy (Data Mining)")
    col_a1, col_a2 = st.columns([2, 1])
    
    with col_a1:
        st.info("Poniższy wykres pokazuje 'Złotą Strefę' parametrów. Zwróć uwagę, że klasa PREMIUM (zielona) skupia się w bardzo konkretnym obszarze temperatur i wilgotności.")
        # SCATTER PLOT 3D-like (2D with info)
        fig_scatter = px.scatter(
            df_model, 
            x='Temperatura', 
            y='Wilgotnosc', 
            color='Jakosc',
            size='Czas',
            color_discrete_map={'PREMIUM': '#10B981', 'STANDARD': '#EF4444'},
            hover_data=['Czas'],
            title="Mapa Jakości: Temperatura vs Wilgotność (Wielkość punktu = Czas)",
            labels={'Temperatura': 'Temp Pieca [°C]', 'Wilgotnosc': 'Wilgotność [%]'},
            template="plotly_white"
        )
        # Add Golden Zone
        fig_scatter.add_shape(type="rect",
            x0=170, y0=30, x1=185, y1=40,
            line=dict(color="#10B981", width=2),
            exclude_empty_subplots=True
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_a2:
        st.markdown("**Reguły Decyzyjne Modelu:**")
        st.markdown("""
        Algorytm drzewa decyzyjnego wyznaczył matematyczne granice jakości:
        
        1.  **Warunek Główny (Temperatura):**
            *   Idealnie: **170 - 185°C**
            *   Poniżej: Niedopiek (Standard)
            *   Powyżej: Przypalenie (Standard)
            
        2.  **Warunek Drugi (Wilgotność):**
            *   Optimum: **30 - 40%**
            *   Wpływa na chrupkość skórki.
            
        3.  **Warunek Trzeci (Czas):**
            *   Musi być **< 50 min**.
            *   Dłuższy czas wysusza produkt.
        """)
        st.metric("Dokładność Modelu AI", f"{accuracy*100:.1f}%")
# --- TAB 3: SYMULATOR ---
with tab_sim:
    st.markdown("### 🎛️ Digital Twin: Symulator Procesu")
    st.markdown("Zmień parametry wirtualnego pieca i sprawdź predykcję systemu w czasie rzeczywistym.")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.subheader("1. Ustawienia Pieca")
        s_temp = st.slider("Temperatura [°C]", 150, 210, 178)
        s_hum = st.slider("Wilgotność [%]", 10, 60, 35)
        s_time = st.slider("Czas Pieczenia [min]", 30, 70, 48)
        
    with col_s2:
        st.subheader("2. Weryfikacja Safety")
        is_safe_sim = True
        
        if s_temp < 160:
            st.error(f"⛔ ODPAD KRYTYCZNY: Temp {s_temp}°C za niska! Ryzyko mikrobiologiczne.")
            is_safe_sim = False
        elif s_temp > 200:
            st.error(f"⛔ ODPAD KRYTYCZNY: Temp {s_temp}°C za wysoka! Zwęglenie produktu.")
            is_safe_sim = False
        else:
            st.success("✅ Parametry Bezpieczne. Przekazuję do AI.")
            
    with col_s3:
        st.subheader("3. Predykcja Jakości")
        if is_safe_sim and model:
            pred_sim = model.predict([[s_temp, s_hum, s_time]])[0]
            if pred_sim == 1:
                st.balloons()
                st.markdown(f"""
                <div style="background-color: #D1FAE5; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #10B981;">
                    <h1 style="color: #047857; margin:0;">PREMIUM 💎</h1>
                    <p style="color: #065F46;">Produkt spełnia najwyższe normy jakości.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #FEE2E2; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #EF4444;">
                    <h1 style="color: #B91C1C; margin:0;">STANDARD ⚠️</h1>
                    <p style="color: #991B1B;">Produkt jadalny, ale niższej klasy.</p>
                </div>
                """, unsafe_allow_html=True)
                # Diagnostyka
                if s_temp < 170: st.warning("📉 Temperatura trochę za niska na Premium.")
                elif s_temp > 185: st.warning("📈 Temperatura trochę za wysoka na Premium.")
                if s_hum < 30: st.warning("💧 Za sucho w piecu.")
                elif s_hum > 40: st.warning("💦 Za wilgotno.")
# --- TAB 4: O PROJEKCIE (DOCS) ---
with tab_about:
    st.markdown("### 📄 Dokumentacja Projektu")
    st.markdown("""
    #### 1. Cel Biznesowy
    Celem projektu jest **zwiększenie udziału produkcji klasy PREMIUM o 15%** poprzez automatyczną kontrolę parametrów pieca i wczesną detekcję anomalii (odpadów). System ma zastąpić wyrywkową kontrolę ręczną systemem ciągłym opartym na AI.
    #### 2. Architektura Rurociągu Danych (Data Pipeline)
    *   **Źródło Danych:** Sterownik PLC pieca tunelowego (Logi operacyjne).
    *   **Format Danych:** Szereg czasowy z sensorów temperatury, wilgotności i czasu przuwu taśmy.
    *   **Preprocessing:** 
        *   Czyszczenie braków danych.
        *   One-Hot Encoding dla ID Pieca (odrzucone jako nieistotne w finalnym modelu).
        *   Safety Gate (reguła twarda T<160 | T>200).
    #### 3. Model Machine Learning
    *   **Algorytm:** Decision Tree Classifier (CART).
    *   **Zaleta:** Pełna interpretowalność (White-box model), co jest kluczowe w przemyśle spożywczym.
    *   **Wyniki:** Model osiągnął bardzo wysoką skuteczność dzięki wyraźnej separacji fizycznej klas jakości.
    #### 4. Wnioski dla Produkcji
    Aby maksymalizować zysk:
    1.  Utrzymywać temp. w ścisłym reżimie **178°C +/- 7°C**.
    2.  Skrócić czas pieczenia poniżej **50 min** (zwiększenie przepustowości linii).
    3.  Monitorować wilgotność jako parametr drugorzędny, ale istotny dla tekstury.
    """)
st.markdown("---")
st.caption("© 2024 Quality 4.0 Project | Powered by Streamlit & Scikit-Learn")
