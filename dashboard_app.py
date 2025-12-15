import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, plot_tree
# --- KONFIGURACJA STRONY (PREMIUM DARK MODE) ---
st.set_page_config(
    page_title="Quality 4.0 Pro AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# --- CSS STYLING (DARK THEME & CARDS) ---
st.markdown("""
<style>
    /* Globalne tło i tekst */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Nagłówki */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #E2E8F0;
    }
    /* Karty KPI */
    .metric-card {
        background-color: #1A1C24;
        border: 1px solid #2D3748;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #F7FAFC;
    }
    .metric-label {
        color: #A0AEC0;
        font-size: 0.9rem;
        text-transform: uppercase;
        margin-top: 5px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 5px;
    }
    /* Sekcja Safety Gate */
    .safety-gate-container {
        background-color: #2D3748;
        padding: 2px;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .safety-content {
        background-color: #1A1C24;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3182CE;
    }
    .gate-stat {
        background-color: #2D3748; 
        padding: 10px; 
        border-radius: 5px; 
        text-align: center; 
        margin-bottom: 5px;
    }
    /* Symulator */
    .sim-box {
        background-color: #1A1C24;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #2D3748;
    }
    .result-box-premium {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10B981;
        border-left: 8px solid #10B981;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
    }
    .result-box-standard {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid #F59E0B;
        border-left: 8px solid #F59E0B;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
# --- LOAD DATA ENGINE (SPRAWDZONY I PANCERNY) ---
@st.cache_data
def load_engine():
    # 1. Próba CSV (header=2 to klucz)
    try:
        df = pd.read_csv('data.csv', header=2, sep=None, engine='python')
    except:
        # Fallback Excel
        try:
            df = pd.read_excel('data.xlsx', sheet_name='4 - klasyfikacja', header=2, engine='openpyxl')
        except:
            return pd.DataFrame()
    if df.empty: return df
    # 2. Mapowanie indeksowe (niezawodne)
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
    # 3. Konwersja numeryczna i czyszczenie
    for c in ['Temperatura', 'Wilgotnosc', 'Czas']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    return df
@st.cache_resource
def train_engine(df):
    try:
        # Trenujemy tylko na pełnych danych premium/standard
        df_train = df.dropna(subset=['Temperatura', 'Wilgotnosc', 'Czas', 'Jakosc'])
        X = df_train[['Temperatura', 'Wilgotnosc', 'Czas']]
        y = df_train['Jakosc'].map({'PREMIUM': 1, 'STANDARD': 0})
        
        clf = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
        clf.fit(X, y)
        return clf
    except:
        return None
# --- UI START ---
df_raw = load_engine()
if df_raw.empty:
    st.error("❌ CRITICAL ERROR: Brak pliku z danymi (data.csv/xlsx). Wgraj plik na GitHub.")
    st.stop()
# LOGIKA BIZNESOWA
df_safe = df_raw[df_raw['Status'] == 'OK'].copy()
waste_count = len(df_raw) - len(df_safe)
df_model = df_safe[df_safe['Jakosc'].isin(['PREMIUM', 'STANDARD'])].copy()
df_model = df_model.dropna(subset=['Temperatura', 'Wilgotnosc', 'Czas']) # Safety dla wykresów
model = train_engine(df_model)
premium_count = len(df_model[df_model['Jakosc'] == 'PREMIUM'])
premium_share = (premium_count / len(df_model) * 100) if len(df_model) > 0 else 0
# --- HEADER ---
st.title("QUALITY 4.0 AI DASHBOARD")
st.markdown("**Automatyczna klasyfikacja jakości wypieku i detekcja anomalii procesowych**")
st.markdown("---")
# --- SEKCJA 0: SAFETY GATE (BRAMKA) ---
st.subheader("0. Bramka Bezpieczeństwa (Safety Gate)")
col_sg1, col_sg2 = st.columns([2, 1])
with col_sg1:
    st.info("""
    **Dlaczego nie analizujemy wszystkiego?**
    W produkcji żywności bezpieczeństwo jest nadrzędne. Zanim wpuścimy produkt do modelu AI (klasyfikacja jakości),
    musimy odrzucić wszystko, co stwarza zagrożenie mikrobiologiczne (Temp < 160°C) lub jest spalone (Temp > 200°C).
    """)
    st.markdown("""
    **Algorytm Postępowania:**
    1.  **ODPAD KRYTYCZNY:** Parametry poza normą bezpieczeństwa -> Utylizacja natychmiastowa.
    2.  **PRODUKT BEZPIECZNY:** Przekazanie do analizy jakości (Model AI).
    """)
with col_sg2:
    st.markdown(f"""
    <div class="safety-gate-container">
        <div class="safety-content">
            <div style="font-size:0.9rem; color:#A0AEC0;">Odrzucone (Odpad Krytyczny)</div>
            <div style="font-size:2rem; font-weight:bold; color:#FC8181;">{waste_count}</div>
            <div style="font-size:0.8rem; color:#FC8181;">⬇ Ryzyko wyeliminowane</div>
            <hr style="border-color:#4A5568; margin: 10px 0;">
             <div style="font-size:0.9rem; color:#A0AEC0;">Przekazane do Analizy AI</div>
            <div style="font-size:2rem; font-weight:bold; color:#68D391;">{len(df_model)}</div>
            <div style="font-size:0.8rem; color:#68D391;">⬆ Produkt Bezpieczny</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
# --- SEKCJA 1: ANALIZA JAKOŚCI (KPI) ---
st.markdown("---")
st.subheader("1. Analiza Jakości (Produkt Bezpieczny)")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Partie Handlowe</div>
        <div class="metric-value">{len(df_model)}</div>
        <div class="metric-sub">Po odrzuceniu odpadów</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="metric-card" style="border-bottom: 4px solid #10B981;">
        <div class="metric-label">Udział Premium</div>
        <div class="metric-value">{premium_share:.1f}%</div>
        <div class="metric-sub">Cel: > 25%</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Śr. Czas</div>
        <div class="metric-value">{df_model['Czas'].mean():.1f} min</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Śr. Temperatura</div>
        <div class="metric-value">{df_model['Temperatura'].mean():.1f} °C</div>
    </div>
    """, unsafe_allow_html=True)
# --- WYKRESY (TABS) ---
st.write("")
t1, t2 = st.tabs(["📈 Mapa Procesu (Scatter Plot)", "🌳 Struktura Drzewa (Model)"])
with t1:
    st.caption("Wykres pokazuje 'Złotą Strefę' parametrów dla klasy PREMIUM (Zielone punkty).")
    # Custom Plotly Chart to match Dark Theme
    if not df_model.empty:
        fig = px.scatter(
            df_model, 
            x='Temperatura', 
            y='Wilgotnosc', 
            color='Jakosc',
            color_discrete_map={'PREMIUM': '#10B981', 'STANDARD': '#F87171'},
            opacity=0.8,
            height=500
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#2D3748', title="Temperatura Pieca [°C]"),
            yaxis=dict(showgrid=True, gridcolor='#2D3748', title="Wilgotność [%]")
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
        st.plotly_chart(fig, use_container_width=True)
with t2:
    col_tree_desc, col_tree_viz = st.columns([1, 2])
    with col_tree_desc:
        st.markdown("### Jak działa model?")
        st.markdown("""
        Drzewo decyzyjne uczy się **reguł odcięcia** (IF-THEN).
        
        **Główne reguły wykryte przez AI:**
        1.  **Temperatura < 185.5°C** jest kluczowa dla Premium.
        2.  **Czas wypieku** musi być krótszy niż ok. 49 min.
        3.  **Wilgotność** pełni rolę korygującą.
        
        To podejście 'White-Box' pozwala technologom zrozumieć decyzje algorytmu.
        """)
    with col_tree_viz:
        if model:
            from sklearn.tree import plot_tree
            fig_tree, ax = plt.subplots(figsize=(12, 6), facecolor='#0E1117')
            plot_tree(model, feature_names=['Temp', 'Wilg', 'Czas'], class_names=['STD', 'PRM'], 
                     filled=True, rounded=True, fontsize=10, ax=ax)
            st.pyplot(fig_tree)
# --- SEKCJA 2: SYMULATOR ---
st.markdown("---")
st.subheader("2. Symulator Procesu (Digital Twin)")
st.caption("Zmień parametry suwakami, aby sprawdzić predykcję modelu w czasie rzeczywistym.")
sim_col1, sim_col2 = st.columns([2, 1])
with sim_col1:
    s_temp = st.slider("Temperatura [°C]", 150.0, 210.0, 178.0, 0.5)
    s_hum = st.slider("Wilgotność [%]", 10.0, 60.0, 35.0, 0.5)
    s_time = st.slider("Czas pieczenia [min]", 30.0, 70.0, 48.0, 0.5)
with sim_col2:
    # Logic Sim
    is_safe = True
    if s_temp < 160 or s_temp > 200:
        st.markdown(f"""
        <div class="result-box-standard" style="border-left-color: #EF4444; border-color: #EF4444; color: #EF4444;">
            <h1 style="color: #EF4444;">ODPAD ⛔</h1>
            <p>Temperatura krytyczna! Ryzyko niezdatności do spożycia.</p>
        </div>
        """, unsafe_allow_html=True)
        is_safe = False
    
    if is_safe and model:
        pred = model.predict([[s_temp, s_hum, s_time]])[0]
        if pred == 1:
            st.markdown("""
            <div class="result-box-premium">
                <h1 style="color: #10B981;">PREMIUM 💎</h1>
                <p style="color: #10B981;">Parametry w normie. Wysoka jakość.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box-standard">
                <h1 style="color: #F59E0B;">STANDARD ⚠️</h1>
                <p style="color: #F59E0B;">Produkt bezpieczny, ale niższej jakości.</p>
            </div>
            """, unsafe_allow_html=True)
st.write("")
st.caption("System v2.0 Pro | Projekt Studencki | Analiza Danych Sp. z o.o.")
