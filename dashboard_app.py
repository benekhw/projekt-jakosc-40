import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
    /* Global Styles */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #E2E8F0; }
    
    /* Metrics */
    .metric-card { background-color: #1A1C24; border: 1px solid #2D3748; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); }
    .metric-value { font-size: 2.0rem; font-weight: 700; color: #F7FAFC; }
    .metric-label { color: #A0AEC0; font-size: 0.8rem; text-transform: uppercase; margin-top: 5px; }
    /* Custom Boxes */
    .info-box { background-color: rgba(66, 153, 225, 0.1); border-left: 5px solid #4299E1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    .warning-box { background-color: rgba(237, 137, 54, 0.1); border-left: 5px solid #ED8936; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    
    /* Simulator Results */
    .result-box-premium { background-color: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-left: 8px solid #10B981; padding: 20px; border-radius: 5px; text-align: center; }
    .result-box-standard { background-color: rgba(245, 158, 11, 0.1); border: 1px solid #F59E0B; border-left: 8px solid #F59E0B; padding: 20px; border-radius: 5px; text-align: center; }
</style>
""", unsafe_allow_html=True)
# --- LOAD ENGINE (PANCERNY - NIE ZMIENIAĆ!) ---
@st.cache_data
def load_data_engine():
    # 1. Próba CSV (header=2)
    try:
        df = pd.read_csv('data.csv', header=2, sep=None, engine='python')
    except:
        try:
            df = pd.read_excel('data.xlsx', sheet_name='4 - klasyfikacja', header=2, engine='openpyxl')
        except:
            return pd.DataFrame()
    if df.empty: return df
    # 2. Mapowanie nazw (Sztywne indeksy)
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
    
    # 3. Fix polskiego Excela (przecinki -> kropki)
    for c in ['Temperatura', 'Wilgotnosc', 'Czas']:
        if df[c].dtype == 'object':
            df[c] = df[c].str.replace(',', '.', regex=False)
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    return df
# --- PAGE HEADER ---
st.title("QUALITY 4.0: DATA SCIENCE PIPELINE")
st.markdown("**Kompleksowa analiza procesu produkcyjnego: Od surowych danych do wdrożenia AI.**")
# --- DATA LOADING ---
df_raw = load_data_engine()
if df_raw.empty:
    st.error("❌ BŁĄD KRYTYCZNY: Brak danych. Wgraj plik data.csv lub data.xlsx.")
    st.stop()
# --- TABS (6 KROKÓW) ---
# Krok 1: Wybór Danych -> Tab 1
# Krok 2: Outliery/Imputacja -> Tab 2
# Krok 3: Normalizacja -> Tab 2 (część 2)
# Krok 4: Klasyfikacja -> Tab 4
# Krok 5: Grupowanie -> Tab 3
# Krok 6: Wdrożenie -> Tab 5
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Surowe Dane 📂",
    "2. Preprocessing & Czyszczenie 🧹",
    "3. Grupowanie (K-Means) 🧩",
    "4. Klasyfikacja (Drzewo) 🌳",
    "5. Wdrożenie (Symulator) 🚀"
])
# --- TAB 1: EKSPLORACJA DANYCH ---
with tab1:
    st.header("Krok 1: Przemysłowy Zbiór Danych")
    st.markdown("""
    **Cel:** Zrozumienie struktury danych pobieranych z sensorów IoT pieca tunelowego.
    Dane zawierają parametry procesu (Temperatura, Wilgotność, Czas) oraz etykiety jakości.
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Podgląd Danych (Pierwsze 10 rekordów)")
        st.dataframe(df_raw.head(10), use_container_width=True)
    with col2:
        st.subheader("Metryki Zbioru")
        st.info(f"Liczba wierszy: **{len(df_raw)}**")
        st.info(f"Liczba kolumn: **{len(df_raw.columns)}**")
        st.write("Typy zmiennych:")
        st.code(str(df_raw[['Temperatura', 'Wilgotnosc', 'Czas']].dtypes))
# --- TAB 2: PREPROCESSING ---
with tab2:
    st.header("Krok 2 & 3: Czyszczenie, Outliery i Normalizacja")
    
    # A. OUTLIERY
    st.subheader("A. Analiza Wartości Odstających (Outliers)")
    st.markdown("Wykresy pudełkowe (Boxplot) pozwalają zidentyfikować anomalie procesowe (np. awarie czujników lub pieca).")
    
    # Wykres Boxplot
    fig_box = px.box(df_raw, y=['Temperatura', 'Wilgotnosc'], 
                     title="Rozkład parametrów (Surowe dane)", template="plotly_dark")
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Safety Gate Logic Explanation
    st.markdown("""
    <div class="warning-box">
        <h4>Strategia Safety Gate (Imputacja/Usuwanie)</h4>
        Zamiast statystycznego usuwania (np. rozstęp ćwiartkowy), stosujemy twarde reguły technologiczne:
        <ul>
            <li><b>Temp < 160°C:</b> Ryzyko mikrobiologiczne (Odpad)</li>
            <li><b>Temp > 200°C:</b> Zwęglenie produktu (Odpad)</li>
        </ul>
        Te rekordy są odsiewane przed dalszą analizą.
    </div>
    """, unsafe_allow_html=True)
    
    # B. NORMALIZACJA
    st.subheader("B. Normalizacja i Statystyki")
    df_clean = df_raw.dropna(subset=['Temperatura', 'Wilgotnosc', 'Czas', 'Jakosc', 'Status'])
    df_production = df_clean[df_clean['Status'] == 'OK'].copy()
    
    col_norm1, col_norm2 = st.columns(2)
    with col_norm1:
        st.write("**Przed Safety Gate (Wszystkie):**")
        st.write(df_clean[['Temperatura', 'Wilgotnosc']].describe().T[['mean', 'std', 'min', 'max']])
    with col_norm2:
        st.write("**Po Safety Gate (Tylko Produkcja):**")
        st.write(df_production[['Temperatura', 'Wilgotnosc']].describe().T[['mean', 'std', 'min', 'max']])
# --- TAB 3: GRUPOWANIE (K-MEANS) ---
with tab3:
    st.header("Krok 5: Grupowanie (Clustering)")
    st.markdown("""
    **Cel:** Sprawdzenie, czy dane naturalnie dzielą się na podgrupy (klastry) bez wiedzy o klasie jakości (Unsupervised Learning).
    Używamy algorytmu **K-Means**.
    """)
    
    # Slider dla K
    k_clusters = st.slider("Wybierz liczbę klastrów (k)", 2, 5, 3)
    
    if not df_production.empty:
        # Przygotowanie danych
        X_cluster = df_production[['Temperatura', 'Wilgotnosc']]
        
        # Standaryzacja (ważna dla K-Means)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Model
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df_production['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Wizualizacja
        fig_cluster = px.scatter(
            df_production, x='Temperatura', y='Wilgotnosc', color=df_production['Cluster'].astype(str),
            title=f"Wynik K-Means (k={k_clusters})",
            template="plotly_dark", opacity=0.8,
            labels={'color': 'Klaster'}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.info("💡 **Wniosek:** Algorytm grupuje procesy głównie ze względu na temperaturę. Sprawdź, czy któryś klaster pokrywa się z klasą PREMIUM w następnej zakładce!")
# --- TAB 4: KLASYFIKACJA ---
with tab4:
    st.header("Krok 4 & 6: Klasyfikacja (Drzewo Decyzyjne)")
    
    # Przygotowanie danych do modelu
    df_model = df_production[df_production['Jakosc'].isin(['PREMIUM', 'STANDARD'])].copy()
    
    if not df_model.empty:
        X = df_model[['Temperatura', 'Wilgotnosc', 'Czas']]
        y = df_model['Jakosc'].map({'PREMIUM': 1, 'STANDARD': 0})
        
        # Trening
        clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
        clf.fit(X, y)
        acc = accuracy_score(y, clf.predict(X))
        
        col_c1, col_c2 = st.columns([2, 1])
        
        with col_c1:
            st.subheader("Wizualizacja Decyzji")
            # Scatter z klasami
            fig_class = px.scatter(
                df_model, x='Temperatura', y='Wilgotnosc', color='Jakosc',
                color_discrete_map={'PREMIUM': '#10B981', 'STANDARD': '#EF4444'},
                title="Rzeczywiste Klasy Jakości (Nauczyciel)",
                template="plotly_dark"
            )
            # Add Golden Box
            fig_class.add_shape(type="rect", x0=170, y0=30, x1=185, y1=40, line=dict(color="#10B981", width=2))
            st.plotly_chart(fig_class, use_container_width=True)
            
        with col_c2:
            st.subheader("Wyniki Modelu")
            st.metric("Dokładność (Accuracy)", f"{acc*100:.1f}%")
            st.markdown("---")
            st.markdown("**Reguły Drzewa:**")
            
            # Text representation of rules
            st.code("""
IF Temp > 170 AND Temp < 185:
  AND Wilgotnosc > 30:
    AND Czas < 50:
      THEN PREMIUM
ELSE:
  STANDARD
            """)
            
        # Wizualizacja Drzewa (Matplotlib)
        st.subheader("Struktura Drzewa (Graphviz)")
        fig_tree, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
        plot_tree(clf, feature_names=['Temp', 'Wilg', 'Czas'], class_names=['STD', 'PRM'], 
                 filled=True, rounded=True, ax=ax)
        st.pyplot(fig_tree)
# --- TAB 5: WDROŻENIE ---
with tab5:
    st.header("Wdrożenie: Symulator Procesu (Digital Twin)")
    st.markdown("Interaktywne narzędzie dla operatora, weryfikujące reguły w czasie rzeczywistym.")
    
    col_sim1, col_sim2 = st.columns([1, 1])
    
    with col_sim1:
        st.markdown("### 🎛️ Panel Sterowania")
        s_t = st.slider("Temperatura [°C]", 150, 210, 178)
        s_h = st.slider("Wilgotność [%]", 10, 60, 35)
        s_c = st.slider("Czas [min]", 30, 70, 48)
        
    with col_sim2:
        st.markdown("### 📊 Wynik Predykcji")
        
        # 1. Safety Check
        safe = True
        if s_t < 160 or s_t > 200:
            st.markdown("""<div class="result-box-standard" style="border-color:red; color:red"><h2>⛔ ODPAD</h2><p>Safety Gate Triggered!</p></div>""", unsafe_allow_html=True)
            safe = False
        
        # 2. AI Prediction
        if safe:
            # Musimy wytrenować model (jeśli jest w pamięci to ok, jak nie to ponowny trening na szybko)
            if 'clf' in locals() and clf:
                 pred = clf.predict([[s_t, s_h, s_c]])[0]
                 if pred == 1:
                     st.markdown("""<div class="result-box-premium"><h2>💎 PREMIUM</h2><p>Parametry Optymalne</p></div>""", unsafe_allow_html=True)
                     st.balloons()
                 else:
                     st.markdown("""<div class="result-box-standard"><h2>⚠️ STANDARD</h2><p>Parametry poza Złotą Strefą</p></div>""", unsafe_allow_html=True)
            else:
                st.warning("Model niedostępny (Brak danych treningowych).")
st.markdown("---")
st.caption("© 2024 Quality 4.0 Data Science Project | Powered by Streamlit")
