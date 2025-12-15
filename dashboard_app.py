import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
st.set_page_config(layout="wide", page_title="Projekt Jakosc 4.0")
# --- STYLE ---
st.markdown("""<style>
.main-header {font-size: 2.5rem; text-align: center; color: #4B5563; margin-bottom: 1rem;} 
.kpi-card {background-color: #F3F4F6; padding: 20px; border-radius: 10px; text-align: center;}
</style>""", unsafe_allow_html=True)
# --- LOAD DATA ---
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
    # Zamiast zgadywać nazwy, bierzemy po prostu kolumny nr 12 i 13
    cols = list(df.columns)
    
    # Mapa nowych nazw (dla pewności nadpisujemy wszystko co ważne)
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
    return df
@st.cache_resource
def train_model(df):
    try:
        X = df[['Temperatura', 'Wilgotnosc', 'Czas']]
        y = df['Jakosc'].map({'PREMIUM': 1, 'STANDARD': 0})
        clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
        clf.fit(X, y)
        return clf
    except:
        return None
# --- UI LOGIC ---
df_raw = load_and_fix_data()
if df_raw.empty:
    st.error("Brak danych (data.csv/xlsx).")
    st.stop()
# Weryfikacja czy kolumny istnieją po zmianie nazw
required = ['Temperatura', 'Wilgotnosc', 'Czas', 'Status', 'Jakosc']
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Nie udalo sie przypisac kolumn: {missing}")
    st.write("Oryginalne kolumny:", list(df_raw.columns)) # Debug
    st.stop()
# Filtrowanie
df_safe = df_raw[df_raw['Status'] == 'OK'].copy()
df_model = df_safe[df_safe['Jakosc'].isin(['PREMIUM', 'STANDARD'])].dropna(subset=['Temperatura', 'Jakosc'])
model = train_model(df_model)
# --- DASHBOARD ---
st.markdown('<div class="main-header">QUALITY 4.0</div>', unsafe_allow_html=True)
# KPI
k1, k2, k3 = st.columns(3)
k1.metric("Partie (OK)", len(df_model))
k2.metric("Odpady", len(df_raw) - len(df_safe), delta="-RYZYKO", delta_color="inverse")
k3.metric("Premium %", f"{(len(df_model[df_model['Jakosc']=='PREMIUM'])/len(df_model)*100):.1f}%")
st.divider()
# Wykresy
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Mapa Jakości")
    fig = px.scatter(df_model, x='Temperatura', y='Wilgotnosc', color='Jakosc', 
                     color_discrete_map={'PREMIUM': 'green', 'STANDARD': 'red'}, opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Symulator")
    t = st.slider("Temp", 150, 210, 180)
    w = st.slider("Wilg", 10, 60, 35)
    c = st.slider("Czas", 30, 70, 48)
    
    if t < 160 or t > 200:
        st.error("ODPAD")
    elif model:
        pred = model.predict([[t, w, c]])[0]
        if pred == 1: st.success("PREMIUM") 
        else: st.warning("STANDARD")
