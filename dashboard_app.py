import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
st.set_page_config(layout="wide", page_title="Projekt Jakosc 4.0")
# --- STYLE ---
st.markdown("""<style>.main-header {font-size: 2.5rem; text-align: center; color: #4B5563; margin-bottom: 1rem;} 
.kpi-card {background-color: #F3F4F6; padding: 20px; border-radius: 10px; text-align: center;}
</style>""", unsafe_allow_html=True)
# --- SIDEBAR ---
with st.sidebar:
    st.title("Panel Kontrolny")
    st.info("System Analizy Jakosci (Wersja Produkcyjna)")
# --- FUNKCJE DANYCH ---
@st.cache_data
def load_data_universal():
    # Najpierw probujemy wczytac CSV (zalecane)
    try:
        return pd.read_csv('data.csv', sep=None, engine='python') # sep=None wykrywa czy przecinek czy srednik
    except:
        pass
    
    # Jak nie ma CSV, probujemy Excela (fallback)
    try:
        return pd.read_excel('data.xlsx', sheet_name='4 - klasyfikacja', header=2, engine='openpyxl')
    except:
        return pd.DataFrame() # Zwracamy pusty jak nic nie zadziala
def clean_data(df):
    # Standaryzacja nazw kolumn
    normalized_cols = {}
    for c in df.columns:
        c_lower = str(c).lower()
        if "temp" in c_lower: normalized_cols[c] = 'Temperatura'
        elif "wilg" in c_lower: normalized_cols[c] = 'Wilgotnosc'
        elif "czas" in c_lower: normalized_cols[c] = 'Czas'
        elif "jakosc" in c_lower or "jakość" in c_lower: normalized_cols[c] = 'Jakosc'
        elif "odpad" in c_lower or "bezpiecz" in c_lower: normalized_cols[c] = 'Status'
    
    df = df.rename(columns=normalized_cols)
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
# --- GLOWNA LOGIKA ---
df_raw = load_data_universal()
if df_raw.empty:
    st.error("Brak danych! Wgraj plik 'data.csv' (zalecane) lub popraw 'data.xlsx' w repozytorium GitHub.")
    st.stop()
# Czyszczenie
df = clean_data(df_raw)
# Sprawdzenie czy mamy wymagane kolumny
required = ['Temperatura', 'Wilgotnosc', 'Czas', 'Jakosc', 'Status']
missing = [c for c in required if c not in df.columns]
if missing:
    st.warning(f"Nie udalo sie automatycznie rozznac kolumn: {missing}")
    st.write("Dostepne kolumny w pliku:", list(df_raw.columns))
    st.stop()
# Safety Gate Logic
n_all = len(df)
df_safe = df[df['Status'] == 'OK'].copy()
n_safe = len(df_safe)
n_waste = n_all - n_safe
# Model Data
df_model = df_safe[df_safe['Jakosc'].isin(['PREMIUM', 'STANDARD'])].dropna(subset=['Temperatura', 'Jakosc'])
model = train_model(df_model)
# --- UI ---
st.markdown('<div class="main-header">QUALITY 4.0: SYSTEM EKSPERCKI</div>', unsafe_allow_html=True)
# 0. BRAMKA
c1, c2 = st.columns([3, 1])
with c1: st.info("BRAMKA BEZPIECZENSTWA: Odsiewanie Odpadow Krytycznych")
with c2: st.metric("Odrzucone", n_waste, delta="- RYZYKO", delta_color="inverse")
# 1. KPI
st.divider()
k1, k2, k3 = st.columns(3)
k1.metric("Partie Handlowe", len(df_model))
k2.metric("Srednia Temp.", f"{df_model['Temperatura'].mean():.1f} C")
k3.metric("Srednia Wilgotnosc", f"{df_model['Wilgotnosc'].mean():.1f} %")
# 2. MODEL
st.divider()
st.subheader("Symulacja Modelu")
sc1, sc2 = st.columns(2)
with sc1:
    t = st.slider("Temperatura", 150, 210, 180)
    w = st.slider("Wilgotnosc", 10, 60, 35)
    c = st.slider("Czas", 30, 70, 48)
with sc2:
    if t < 160 or t > 200:
        st.error("ODPAD KRYTYCZNY (Temperatura poza norma)")
    elif model:
        res = model.predict([[t, w, c]])[0]
        if res == 1: st.success("PREMIUM")
        else: st.warning("STANDARD")
st.caption("Dane zaladowane poprawnie.")
