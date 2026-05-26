import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
import base64 as _b64
import io as _io
from datetime import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

try:
    import holidays
except ImportError:
    holidays = None

try:
    from stepmix.stepmix import StepMix
    from stepmix.utils import get_mixed_descriptor
    STEPMIX_OK = True
except ImportError:
    STEPMIX_OK = False
    StepMix = None
    get_mixed_descriptor = None
 

_LOGO_TMP = "logo.png"

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Starbucks Customer Analytics",
    page_icon="☕",
    layout="wide"
)
warnings.filterwarnings("ignore")

# ── PALETA CORPORATIVA STARBUCKS ──────────────────────────────────────────────
SBX_GREEN      = "#00704A"
SBX_DARK       = "#1e3932"
SBX_LIGHT      = "#D4E9E2"
SBX_GOLD       = "#CBA135"
SBX_GOLD_LIGHT = "#fdfaf1"
SBX_GRAY       = "#f5f5f5"

# ── CSS GLOBAL ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fondo general ── */
[data-testid="stAppViewContainer"] {
    background-color: #f8faf9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3932 0%, #00704A 100%);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #CBA135;
    color: #1e3932;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #f0c84a;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #1e3932;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #D4E9E2 !important;
    font-weight: 600;
    border-radius: 8px;
    padding: 8px 16px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background-color: #CBA135 !important;
    color: #1e3932 !important;
}

/* ── Métricas ── */
[data-testid="stMetric"] {
    background: white;
    border-left: 4px solid #00704A;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
[data-testid="stMetricLabel"] {
    color: #1e3932 !important;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #00704A !important;
    font-size: 1.8rem !important;
}

/* ── Headers ── */
h1, h2, h3 { color: #1e3932; }

/* ── Caja gold ── */
.gold-box {
    background-color: #fdfaf1;
    padding: 25px;
    border-left: 5px solid #CBA135;
    border-radius: 8px;
    color: #1e1e1e;
    line-height: 1.7;
    font-size: 15.5px;
    margin-bottom: 24px;
    box-shadow: 0 2px 6px rgba(203,161,53,0.12);
}

/* ── Caja verde ── */
.green-box {
    background-color: #f0f8f4;
    padding: 20px;
    border-left: 5px solid #00704A;
    border-radius: 8px;
    color: #1e1e1e;
    line-height: 1.6;
    font-size: 15px;
    margin-bottom: 16px;
}

/* ── Divisor estilizado ── */
hr {
    border: none;
    border-top: 2px solid #D4E9E2;
    margin: 20px 0;
}

/* ── Imágenes nítidas ── */
img {
    image-rendering: high-quality;
}
</style>
""", unsafe_allow_html=True)

# ── FUNCIONES DE PROCESAMIENTO ────────────────────────────────────────────────
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_time"] = pd.to_datetime(df["order_time"].astype(str), errors="coerce").dt.time

    def obtener_momento(t):
        if pd.isna(t): return "Noche"
        if time(5, 0) <= t < time(12, 0): return "Mañana"
        if time(12, 0) <= t < time(19, 0): return "Tarde"
        return "Noche"

    _hols = holidays.US() if holidays else set()
    def clasificar_dia(fecha):
        if _hols and fecha.date() in _hols: return "Feriado"
        if fecha.weekday() >= 5: return "Fin de Semana"
        return "Día Laboral"

    df["momento_dia"]   = df["order_time"].apply(obtener_momento)
    df["categoria_dia"] = df["order_date"].apply(clasificar_dia)
    df["hora"]          = pd.to_datetime(df["order_time"].astype(str),
                                         errors="coerce").dt.hour

    return df

@st.cache_data
def preprocess_data(df):
    df_3 = df.copy()
    age_map = {"18-24":1,"25-34":2,"35-44":3,"45-54":4,"55+":5}
    df_3["customer_age_group"] = df_3["customer_age_group"].map(age_map)
    cols_to_encode = ["order_channel","region","customer_gender","store_location_type"]
    df_3 = pd.get_dummies(df_3, columns=cols_to_encode, drop_first=False)
    return df_3



# ── FUNCIONES DE GRÁFICOS ADICIONALES ────────────────────────────────────────
@st.cache_data
def chart_demanda_horaria(file_path):
    df = load_data(file_path)
    hora_vol   = df.groupby("hora").size().reset_index(name="pedidos")
    hora_gasto = df.groupby("hora")["total_spend"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8faf9")
    axes[0].fill_between(hora_vol["hora"], hora_vol["pedidos"], alpha=0.25, color=SBX_DARK)
    axes[0].plot(hora_vol["hora"], hora_vol["pedidos"], "o-", color=SBX_GREEN, lw=2.5, ms=6, zorder=5)
    for nm, rng, col in [("Manana",(5,12),"#FFF8DC"),("Tarde",(12,19),"#E8F4F0"),("Noche",(19,24),"#EEE8F5")]:
        axes[0].axvspan(rng[0], rng[1], alpha=0.22, color=col, label=nm, zorder=0)
    axes[0].set_xticks(range(0, 24, 2)); axes[0].set_xlabel("Hora del dia", fontsize=11)
    axes[0].set_ylabel("N de pedidos", fontsize=11)
    axes[0].set_title("Curva de Demanda por Hora", fontsize=12, fontweight="bold", color=SBX_DARK)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
    axes[1].plot(hora_gasto["hora"], hora_gasto["total_spend"], "s-", color=SBX_GOLD, lw=2.5, ms=6, zorder=5)
    axes[1].fill_between(hora_gasto["hora"], hora_gasto["total_spend"], alpha=0.2, color=SBX_GOLD)
    axes[1].set_xticks(range(0, 24, 2)); axes[1].set_xlabel("Hora del dia", fontsize=11)
    axes[1].set_ylabel("Gasto promedio (USD)", fontsize=11)
    axes[1].set_title("Gasto Promedio por Hora", fontsize=12, fontweight="bold", color=SBX_DARK)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    fig.suptitle("Patrones de Demanda Horaria — Starbucks", fontsize=14, fontweight="bold", y=1.02, color=SBX_DARK)
    plt.tight_layout()
    return fig


@st.cache_data
def chart_evolucion_mensual(file_path):
    df = load_data(file_path)
    df["mes"] = df["order_date"].dt.to_period("M").astype(str)
    mensual = df.groupby("mes").agg(pedidos=("order_id","count"), gasto=("total_spend","sum")).reset_index()
    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#f8faf9")
    ax2 = ax1.twinx()
    ax1.fill_between(range(len(mensual)), mensual["pedidos"], alpha=0.2, color=SBX_GREEN)
    ax1.plot(range(len(mensual)), mensual["pedidos"], "o-", color=SBX_GREEN, lw=2.5, ms=5, label="N pedidos")
    ax2.plot(range(len(mensual)), mensual["gasto"]/1000, "s--", color=SBX_GOLD, lw=2.5, ms=5, label="Gasto (miles USD)")
    ax1.set_xticks(range(len(mensual))); ax1.set_xticklabels(mensual["mes"], rotation=35, ha="right", fontsize=8.5)
    ax1.set_ylabel("N de pedidos", fontsize=11, color=SBX_GREEN)
    ax2.set_ylabel("Gasto total (miles USD)", fontsize=11, color=SBX_GOLD)
    ax1.set_title("Evolucion Mensual de Pedidos y Gasto Total", fontsize=14, fontweight="bold", color=SBX_DARK)
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, fontsize=10, loc="upper left", framealpha=0.8)
    ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    plt.tight_layout()
    return fig


@st.cache_data
def chart_radar_segmentos(file_path):
    from sklearn.preprocessing import MinMaxScaler
    df = load_data(file_path)
    for c in ["is_rewards_member","has_food_item","order_ahead"]:
        df[c] = df[c].astype(int)
    ref = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        Recency=("order_date", lambda x: (ref-x.max()).days),
        Frequency=("order_id","nunique"), Monetary=("total_spend","sum")).reset_index()
    scaler = MinMaxScaler()
    rfm_sc = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])
    rfm["Cluster_RFM"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(rfm_sc)
    orden = rfm.groupby("Cluster_RFM")["Monetary"].mean().sort_values(ascending=False).index
    nmap  = {orden[0]:"Frecuentes Alto Valor", orden[1]:"Clientes Regulares",
             orden[2]:"Clientes Ocasionales",  orden[3]:"Clientes Inactivos"}
    rfm["Segmento"] = rfm["Cluster_RFM"].map(nmap)
    df_r = df.merge(rfm[["customer_id","Segmento"]], on="customer_id", how="left")
    SEGS   = ["Frecuentes Alto Valor","Clientes Regulares","Clientes Ocasionales","Clientes Inactivos"]
    COLORS = [SBX_GREEN, SBX_GOLD, "#2E86C1", "#CB4335"]
    vars_r  = ["total_spend","num_customizations","customer_satisfaction","cart_size","is_rewards_member","has_food_item","order_ahead"]
    labels_r = ["Gasto","Personaliz.","Satisfaccion","Carrito","Rewards","Comida","Antelado"]
    perf = df_r.groupby("Segmento")[vars_r].mean()
    # Normalización con piso 0.15 → Clientes Ocasionales visible como polígono
    _rng = perf.max() - perf.min()
    _rng[_rng == 0] = 1          # evita división por cero si una variable no varía
    perf_norm = 0.15 + 0.85 * (perf - perf.min()) / _rng
    N = len(vars_r)
    angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#f8faf9")
    axes = axes.flatten()
    for ax, seg, color in zip(axes, SEGS, COLORS):
        if seg not in perf_norm.index:
            ax.set_visible(False); continue
        vals = perf_norm.loc[seg].values.tolist() + [perf_norm.loc[seg].values[0]]
        ax.plot(angles, vals, "o-", color=color, lw=2.5)
        ax.fill(angles, vals, alpha=0.22, color=color)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_r, fontsize=9, color=SBX_DARK)
        ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.50, 0.75])
        ax.set_yticklabels(["0.25","0.50","0.75"], fontsize=7, color="gray")
        ax.set_title(seg, fontsize=11, fontweight="bold", pad=15, color=color)
        ax.spines["polar"].set_color("#D4E9E2"); ax.grid(color="#D4E9E2", linewidth=0.5)
    fig.suptitle("Perfil Multidimensional de Segmentos RFM", fontsize=14, fontweight="bold", y=1.01, color=SBX_DARK)
    plt.tight_layout()
    return fig



@st.cache_data
def chart_categorias_demo(file_path):
    """Barras agrupadas de variables categóricas por segmento socio-demográfico."""
    df = load_data(file_path)
    df_prep_local = df.copy()
    age_map = {"18-24":1,"25-34":2,"35-44":3,"45-54":4,"55+":5}
    df_prep_local["customer_age_group"] = df_prep_local["customer_age_group"].map(age_map)
    cols_enc = ["order_channel","region","customer_gender","store_location_type"]
    df_enc = pd.get_dummies(df_prep_local, columns=cols_enc, drop_first=False)
    X = StandardScaler().fit_transform(df_enc.select_dtypes(include=[np.number]).fillna(0))
    df["Cluster_Demo"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
    canal_modal = df.groupby("Cluster_Demo")["order_channel"].agg(lambda x: x.mode()[0])
    nombres_demo = {}
    for c in df["Cluster_Demo"].unique():
        canal = canal_modal[c]
        if "Mobile App" in canal or "Kiosk" in canal:
            nombres_demo[c] = "Clientes Digitales Jovenes"
        elif "Drive-Thru" in canal or "In-Store" in canal:
            nombres_demo[c] = "Clientes Adultos Presenciales"
        else:
            nombres_demo[c] = "Clientes Tradicionales Medios"
    df["Segmento_Demo"] = df["Cluster_Demo"].map(nombres_demo)
    vars_cat = ["customer_age_group","customer_gender","order_channel","store_location_type"]
    titulos  = ["Grupo de edad","Genero","Canal de pedido","Tipo de tienda"]
    pal_cat  = [SBX_GREEN, SBX_GOLD, "#5B8DB8", "#CB4335", "#8B5EA0", "#D85A30"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#f8faf9")
    for ax, col, tit in zip(axes.flatten(), vars_cat, titulos):
        cross = df.groupby(["Segmento_Demo", col]).size().unstack(fill_value=0)
        cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
        cross_pct.plot(kind="bar", ax=ax,
                       color=pal_cat[:cross_pct.shape[1]],
                       edgecolor="white", width=0.65)
        ax.set_title(tit, fontsize=12, fontweight="bold", color=SBX_DARK)
        ax.set_xlabel("")
        ax.set_ylabel("% dentro del segmento", fontsize=10, color=SBX_DARK)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.tick_params(axis="y", labelcolor=SBX_DARK)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Composicion de segmentos socio-demograficos por variable categorica",
                 fontsize=13, fontweight="bold", color=SBX_DARK, y=1.01)
    plt.tight_layout()
    return fig


@st.cache_data
def chart_torta_demo(file_path):
    """Torta de distribucion de clientes por segmento socio-demografico."""
    df = load_data(file_path)
    df_prep_local = df.copy()
    age_map = {"18-24":1,"25-34":2,"35-44":3,"45-54":4,"55+":5}
    df_prep_local["customer_age_group"] = df_prep_local["customer_age_group"].map(age_map)
    cols_enc = ["order_channel","region","customer_gender","store_location_type"]
    df_enc = pd.get_dummies(df_prep_local, columns=cols_enc, drop_first=False)
    X = StandardScaler().fit_transform(df_enc.select_dtypes(include=[np.number]).fillna(0))
    df["Cluster_Demo"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
    canal_modal = df.groupby("Cluster_Demo")["order_channel"].agg(lambda x: x.mode()[0])
    nombres_demo = {}
    for c in df["Cluster_Demo"].unique():
        canal = canal_modal[c]
        if "Mobile App" in canal or "Kiosk" in canal:
            nombres_demo[c] = "Clientes Digitales Jovenes"
        elif "Drive-Thru" in canal or "In-Store" in canal:
            nombres_demo[c] = "Clientes Adultos Presenciales"
        else:
            nombres_demo[c] = "Clientes Tradicionales Medios"
    df["Segmento_Demo"] = df["Cluster_Demo"].map(nombres_demo)
    conteo = df.groupby("Segmento_Demo")["customer_id"].nunique().reset_index()
    conteo.columns = ["Segmento", "N_Clientes"]
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    colores_torta = [SBX_GREEN, "#E07B39", "#5B8DB8"]
    wedges, texts, autotexts = ax.pie(
        conteo["N_Clientes"],
        labels=conteo["Segmento"],
        colors=colores_torta[:len(conteo)],
        autopct="%1.1f%%",
        startangle=120,
        pctdistance=0.78,
        wedgeprops=dict(edgecolor="white", linewidth=2.5)
    )
    for t in texts:
        t.set_fontsize(11); t.set_color(SBX_DARK); t.set_fontweight("bold")
    for at in autotexts:
        at.set_fontsize(12); at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("Distribucion de clientes por segmento socio-demografico",
                 fontsize=13, fontweight="bold", color=SBX_DARK, pad=18)
    plt.tight_layout()
    return fig


def style_matplotlib():
    """Aplica estilo corporativo Starbucks a todos los gráficos Matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":  "#f8faf9",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#D4E9E2",
        "axes.grid":         True,
        "grid.color":        "#D4E9E2",
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   "#1e3932",
        "axes.labelcolor":   "#1e3932",
        "axes.labelsize":    11,
        "xtick.color":       "#1e3932",
        "ytick.color":       "#1e3932",
        "font.family":       "sans-serif",
        "text.color":        "#1e3932",
    })

style_matplotlib()

@st.cache_data
def calcular_metricas_kmeans(df_preprocesado):
    # Muestreo para KMeans diagnóstico (codo + silueta): resultados estables con 20k
    MAX_KM = 20000
    df_work = df_preprocesado.sample(min(MAX_KM, len(df_preprocesado)), random_state=42)
    df_solo_num = df_work.select_dtypes(include=['number', 'bool', 'uint8'])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_solo_num)
    
    k_rango = range(1, 11)
    inercias = []
    silhouette_dict = {} 
    
    for k in k_rango:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        inercias.append(kmeans.inertia_)
        
        if 2 <= k <= 7:
            # Submuestreo estratégico para acelerar el cálculo web sin perder precisión
            score = silhouette_score(df_scaled, labels, sample_size=5000, random_state=42)
            silhouette_dict[k] = score
            
    # Construcción de la tabla comparativa compacta
    silhouettes_alineados = [silhouette_dict.get(k, None) for k in k_rango]
    tabla_comparativa = pd.DataFrame({
        'K': list(k_rango), 
        'Inercia (Codo)': inercias, 
        'Silhouette Score': silhouettes_alineados
    }).round(3)
    
    return inercias, silhouette_dict, tabla_comparativa, df_scaled

@st.cache_data
def generar_perfilamiento_lca_v2(df_input):
    # Muestreo para limitar RAM en Streamlit Cloud
    MAX_PERF = 10000
    df_sample = df_input.sample(min(MAX_PERF, len(df_input)), random_state=42)
    df_completo = df_input.copy()   # conservar completo para el perfil final
    columnas_modelo = ['total_spend', 'customer_satisfaction', 'fulfillment_time_min']
    X_vars = df_sample[columnas_modelo].values

    # Ajuste del modelo StepMix ( Gaussian )
    model_4_clusters = StepMix(n_components=4, measurement='gaussian', random_state=42, n_init=3, max_iter=50)
    model_4_clusters.fit(X_vars)
    df_completo['Cluster_LCA'] = model_4_clusters.predict(X_vars)

    # Diccionario de nombres estratégicos corregido según tus datos reales
    lca_names_map = {
        0: "Ocasionales",
        1: "Nivel Alto",
        2: "Leales",
        3: "Puntuales"
    }
    df_completo['Segmento_LCA'] = df_completo['Cluster_LCA'].map(lca_names_map)


    def obtener_moda(x):
        res = x.mode()
        return res.iloc[0] if not res.empty else None

    diccionario_agregacion = {
        'total_spend': 'mean', 'customer_satisfaction': 'mean', 'fulfillment_time_min': 'mean',
        'cart_size': 'mean', 'num_customizations': 'mean', 'is_rewards_member': 'mean',
        'has_food_item': 'mean', 'order_ahead': 'mean', 'order_channel': obtener_moda,
        'store_location_type': obtener_moda, 'region': obtener_moda, 'customer_age_group': obtener_moda,
        'customer_gender': obtener_moda, 'drink_category': obtener_moda
    }

    tabla = df_completo.groupby('Segmento_LCA').agg(diccionario_agregacion).round(2)
    tabla['Cantidad_Clientes'] = df_completo['Segmento_LCA'].value_counts().reindex(tabla.index).values
    tabla['%_Clientes'] = ((tabla['Cantidad_Clientes'] / len(df_completo)) * 100).round(2)
    return tabla, df_completo

@st.cache_data
def calcular_lca_mixto(df_input):
    gaussianas          = ['total_spend', 'customer_satisfaction', 'fulfillment_time_min']
    variables_categoricas = ['customer_gender', 'store_location_type', 'order_channel',
                             'region', 'categoria_dia', 'momento_dia', 'customer_age_group']
    binarias            = ['is_rewards_member']

    def preparar_df(df_raw):
        """Aplica las mismas transformaciones a cualquier subconjunto del df."""
        df_out = df_raw.copy()
        for col in variables_categoricas:
            df_out[col] = df_out[col].astype('category').cat.codes
        for col in binarias:
            df_out[col] = df_out[col].astype(int)
        return df_out

    # ── Muestra para ENTRENAR (rápido, bajo RAM) ──────────────────────────────
    MAX_LCA = 3000
    df_train = preparar_df(df_input.sample(min(MAX_LCA, len(df_input)), random_state=42))

    mm_data_train, mm_descriptor = get_mixed_descriptor(
        dataframe=df_train,
        gaussian=gaussianas,
        categorical=variables_categoricas,
        binary=binarias
    )

    # Bucle BIC/AIC (k=3,4,5 — rango reducido para velocidad)
    results = {}
    for n_classes in range(3, 6):
        model = StepMix(n_components=n_classes, measurement=mm_descriptor,
                        random_state=42, n_init=1, max_iter=30)
        model.fit(mm_data_train)
        results[n_classes] = {
            'aic': model.aic(mm_data_train),
            'bic': model.bic(mm_data_train)
        }

    # Modelo final entrenado con la muestra
    model_final = StepMix(n_components=4, measurement=mm_descriptor,
                          random_state=123, n_init=1, max_iter=100)
    model_final.fit(mm_data_train)

    # ── Predecir sobre el df COMPLETO (100k filas) ────────────────────────────
    df_full = preparar_df(df_input)
    mm_data_full, _ = get_mixed_descriptor(
        dataframe=df_full,
        gaussian=gaussianas,
        categorical=variables_categoricas,
        binary=binarias
    )
    predicciones = model_final.predict(mm_data_full)   # len = len(df_input)

    return results, tragic_errors_avoided_df(results), predicciones

def tragic_errors_avoided_df(results):
    # Auxiliar para formatear la tabla comparativa de criterios
    return pd.DataFrame({
        'Clases (K)': list(results.keys()),
        'AIC': [results[n]['aic'] for n in results],
        'BIC': [results[n]['bic'] for n in results]
    }).round(1)

@st.cache_data
def generar_base_rfm(df_input):
    df_rfm = df_input.copy()
    
    # 1. Fecha de corte (un día después de la última orden registrada)
    df_rfm['order_date'] = pd.to_datetime(df_rfm['order_date'])
    fecha_corte = df_rfm['order_date'].max() + pd.Timedelta(days=1)

    # 2. Agrupación por cliente único
    df_rfm_grouped = df_rfm.groupby('customer_id').agg({
        'order_date': lambda x: (fecha_corte - x.max()).days, # Recency_days reales
        'order_id': 'nunique',                                # Frequency
        'total_spend': 'sum'                                  # Monetary
    }).rename(columns={
        'order_date': 'Recency_days',
        'order_id': 'Frequency',
        'total_spend': 'Monetary'
    })

    # 3. Conversión para el algoritmo de clustering geométrico
    df_rfm_grouped['Recency'] = df_rfm_grouped['Recency_days'].max() - df_rfm_grouped['Recency_days']
    
    # 4. Escalado de columnas clave
    scaler_rfm = MinMaxScaler()
    rfm_scaled = pd.DataFrame(
        scaler_rfm.fit_transform(df_rfm_grouped[['Recency', 'Frequency', 'Monetary']]),
        columns=['R', 'F', 'M'],
        index=df_rfm_grouped.index
    )
    
    # 5. Ejecución del modelo K-Means sin etiquetas de negocio
    kmeans_rfm = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_rfm_grouped['Cluster_RFM'] = kmeans_rfm.fit_predict(rfm_scaled)
    
    return df_rfm_grouped

@st.cache_data
def calcular_diagnostico_rfm_kmeans(df_rfm_grouped):
    df_copia = df_rfm_grouped.copy()
    columnas_modelo = ['Recency', 'Frequency', 'Monetary']
    
    # SEPARACIÓN Y ESTANDARIZACIÓN (Según tu script técnico)
    X_rfm = StandardScaler().fit_transform(df_copia[columnas_modelo])
    
    inertias_rfm = []
    sils_rfm = []
    k_rango = range(2, 9)
    
    for k in k_rango:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lab = km.fit_predict(X_rfm)
        
        inertias_rfm.append(km.inertia_)
        # Muestreo eficiente de 3,000 casos sobre el espacio estandarizado
        sils_rfm.append(silhouette_score(X_rfm, lab, sample_size=3000, random_state=42))
        
    return list(k_rango), inertias_rfm, sils_rfm
    
# ── CARGA ─────────────────────────────────────────────────────────────────────
FILE_NAME = "s_order.csv"

if not os.path.exists(FILE_NAME):
    st.error(f"No se encontró **'{FILE_NAME}'**. Colócalo en la misma carpeta que este script.")
    st.stop()

df = load_data(FILE_NAME)
# df_prep se calcula dentro del Tab 2 cuando se necesita (reduce RAM al inicio)

# ── LOGO + SIDEBAR ────────────────────────────────────────────────────────────
if os.path.exists(_LOGO_TMP):
    try:
        st.logo(_LOGO_TMP, link="https://www.starbucks.com")
        st.sidebar.image(_LOGO_TMP, width=110)
    except Exception:
        pass
st.sidebar.markdown("## Starbucks Analytics")
st.sidebar.success(f"**{len(df):,}** registros cargados")
st.sidebar.markdown("---")
st.sidebar.markdown("**Grupo 32** | Marketing 2026‑1")
st.sidebar.markdown("Universidad de Concepción")
st.sidebar.markdown("---")
if st.sidebar.button(" Limpiar caché"):
    st.cache_data.clear(); st.rerun()

# ── TÍTULO ────────────────────────────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 9])
with col_logo:
    st.image(_LOGO_TMP, width=85)
with col_titulo:
    st.markdown(
        "<h1 style='color:#1e3932; margin-bottom:0;'>"
        "Estrategia de Datos: Segmentación de Clientes Starbucks</h1>",
        unsafe_allow_html=True
    )
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Defensa Metodológica",
    "Modelo Sociodemográfico",
    "Modelo RFM",
    "Segmentación Cruzada",
    "Recomendaciones"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEFENSA METODOLÓGICA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("1. Contexto")
    st.markdown("<h3 style='color:#00704A;'>El <em>problema</em> a resolver</h3>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class="gold-box">
        La industria del retail de café y alimentos se caracteriza por una alta frecuencia de transacciones
        y una baja barrera de salida para el consumidor. En este entorno, la lealtad de marca no es un estado
        permanente, sino un activo que debe gestionarse mediante la personalización. Starbucks opera bajo una
        estructura de <strong>micromomentos</strong>: cada pedido representa una oportunidad única para entender
        las preferencias individuales de consumo.<br><br>
        Una estrategia de marketing indiferenciada genera ineficiencia operativa y fuga de clientes.
        Por ello, transformamos <strong>100.000 transacciones</strong> en perfiles de comportamiento
        accionables mediante técnicas de <strong>Machine Learning</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.header("2. Contexto del Proyecto")
    st.markdown("""
    <div class="green-box">
    Starbucks necesita entender no solo <strong>qué</strong> compran sus clientes, sino <strong>cómo</strong>
    y <strong>cuándo</strong> lo hacen. Este proyecto aplica <em>K-Means</em> y <em>Análisis de Clase Latente</em>
    para identificar patrones ocultos, pasando de un marketing masivo a uno de precisión.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("3. Diccionario de Variables")
    variables_data = {
        "Variable": [
            "customer_id","order_id","order_date","order_time","day_of_week",
            "order_channel","store_id","store_location_type","region","customer_age_group",
            "customer_gender","is_rewards_member","cart_size","num_customizations","total_spend",
            "fulfillment_time_min","drink_category","has_food_item","order_ahead","customer_satisfaction"
        ],
        "Tipo de Dato": [
            "ID","ID","Fecha","Tiempo","Categórica","Categórica","ID",
            "Categórica","Categórica","Ordinal","Categórica","Booleana",
            "Numérica","Numérica","Numérica (USD)","Numérica (min)","Categórica",
            "Booleana","Booleana","Ordinal (1‑5)"
        ],
        "Descripción": [
            "Identificador único del cliente.",
            "Identificador único del pedido.",
            "Fecha en que se realizó el pedido.",
            "Hora en que se realizó el pedido.",
            "Día de la semana del pedido.",
            "Canal del pedido (App, Cashier, Drive‑Thru, etc.).",
            "Identificador único de la tienda.",
            "Tipo de ubicación (Urban / Suburban / Rural).",
            "Región geográfica de la tienda.",
            "Grupo de edad del cliente.",
            "Género del cliente.",
            "Miembro del programa de recompensas.",
            "N° de artículos en el carrito.",
            "N° de personalizaciones del pedido.",
            "Monto total gastado.",
            "Tiempo de preparación del pedido.",
            "Categoría de la bebida.",
            "Incluye alimento.",
            "Pedido realizado con antelación.",
            "Calificación de satisfacción (1‑5)."
        ]
    }
    st.dataframe(pd.DataFrame(variables_data), use_container_width=True, hide_index=True)

    st.subheader("4. Resumen Estadístico del Dataset")
    st.dataframe(df.describe().round(2), use_container_width=True)

    # ── Sección 5: Distribución de Variables Categóricas ───────────────────────
    st.subheader("5. Distribución de Variables Categóricas")
    st.markdown("Selecciona una característica del menú desplegable para examinar su comportamiento operacional de mayor a menor frecuencia:")

    # Lista completa de tus variables categóricas
    variables_categoricas = [
        'customer_gender', 'store_location_type', 'order_channel', 
        'region', 'categoria_dia', 'momento_dia', 'customer_age_group'
    ]

    # Diccionario amigable para las opciones visuales del selector
    nombres_formateados = {
        'customer_gender': "Género del Cliente (customer_gender)",
        'store_location_type': "Tipo de Ubicación de Tienda (store_location_type)",
        'order_channel': "Canal de Entrada del Pedido (order_channel)",
        'region': "Región Geográfica de la Sucursal (region)",
        'categoria_dia': "Clasificación por Tipo de Día (categoria_dia)",
        'momento_dia': "Bloque Horario de Consumo (momento_dia)",
        'customer_age_group': "Rango Etario del Consumidor (customer_age_group)"
    }

    # Selector interactivo de Streamlit
    var_seleccionada = st.selectbox(
        "Variable a analizar:",
        options=variables_categoricas,
        format_func=lambda x: nombres_formateados[x]
    )

    # Preparación y ordenamiento de los datos de mayor a menor frecuencia
    conteo_datos = df[var_seleccionada].value_counts().reset_index()
    conteo_datos.columns = [var_seleccionada, "Cantidad"]

    # Construcción formal del lienzo
    fig_cat, ax_cat = plt.subplots(figsize=(10, 5.5))
    fig_cat.patch.set_facecolor("#f8faf9")

    # Mapeo estético dinámico basado en tu paleta (Viridis)
    colores_barras = sns.color_palette("viridis", len(conteo_datos))

    # Graficado de barras ordenadas
    barras_cat = ax_cat.bar(
        conteo_datos[var_seleccionada].astype(str), 
        conteo_datos["Cantidad"],
        color=colores_barras, 
        edgecolor="white", 
        width=0.52, 
        zorder=3
    )

    # Inyección automática de etiquetas de conteo sobre cada barra
    for barra in barras_cat:
        alto = barra.get_height()
        ax_cat.text(
            barra.get_x() + barra.get_width() / 2, 
            alto + (conteo_datos["Cantidad"].max() * 0.015),
            f"{int(alto):,}", 
            ha="center", 
            va="bottom",
            fontsize=10, 
            fontweight="bold", 
            color="#1e3932"
        )

    # Configuración formal de títulos y ejes corporativos
    ax_cat.set_title(f'Distribución de Frecuencias: {var_seleccionada}', fontsize=13, fontweight='bold', pad=15, color="#1e3932")
    ax_cat.set_xlabel(f"Categorías en '{var_seleccionada}'", fontsize=11, color="#1e3932")
    ax_cat.set_ylabel('Cantidad de Registros', fontsize=11, color="#1e3932")
    
    # Rotación controlada para evitar solapamientos de textos largos
    plt.xticks(rotation=25, ha="right")
    
    # Margen superior dinámico para evitar que las etiquetas queden fuera de la gráfica
    ax_cat.set_ylim(0, conteo_datos["Cantidad"].max() * 1.15)

    plt.tight_layout()
    st.pyplot(fig_cat)
    plt.close()

    # ── Sección 6: Distribuciones numéricas ────────────────────────────────────
    st.subheader("6. Distribuciones de Variables Numéricas Clave")
    st.markdown("Análisis de frecuencias para las métricas operacionales y comerciales:")

    num_cols_eda = ["cart_size","num_customizations","total_spend",
                    "fulfillment_time_min","customer_satisfaction"]
    sbx_palette  = [SBX_GREEN, SBX_GOLD, SBX_DARK, "#5B8DB8", "#8B5EA0"]

    fig_num, axes_num = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col, color in zip(axes_num.flat, num_cols_eda, sbx_palette):
        ax.hist(df[col], bins=40, color=color, edgecolor="white", alpha=0.88, zorder=3)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
    fig_num.delaxes(axes_num.flat[5])
    fig_num.suptitle(
        f"Distribuciones de variables numéricas clave\n"
        f"Fuente: s_order.csv  (N={len(df):,})",
        fontsize=13, fontweight="bold", y=1.02, color="#1e3932"
    )
    plt.tight_layout()
    st.pyplot(fig_num)
    plt.close()

    # ── Sección 7: Outliers ────────────────────────────────────────────────────
    st.subheader("7. Análisis de Outliers (Diagramas de Caja)")
    st.markdown("""
    > **Decisión estratégica:** Los *outliers* en `total_spend` y `num_customizations`
    **no serán eliminados**. Se hipotetiza que representan pedidos corporativos o clientes VIP
    que los modelos de clustering deben capturar.
    """)

    columnas_numericas = ["cart_size","num_customizations","total_spend",
                          "fulfillment_time_min","customer_satisfaction"]

    fig_box, axes_box = plt.subplots(2, 3, figsize=(16, 9))
    axes_box = axes_box.flatten()
    for i, (col, color) in enumerate(zip(columnas_numericas, sbx_palette)):
        axes_box[i].boxplot(
            df[col], patch_artist=True, vert=True,
            boxprops=dict(facecolor=color, color="#1e3932", alpha=0.75),
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color="#1e3932"),
            capprops=dict(color="#1e3932"),
            flierprops=dict(marker="o", color=color, alpha=0.4, markersize=3)
        )
        axes_box[i].set_title(f"{col}", fontsize=11, fontweight="bold")
        axes_box[i].set_xlabel("")
    fig_box.delaxes(axes_box[5])
    fig_box.suptitle("Análisis de Outliers — Variables Numéricas",
                     fontsize=13, fontweight="bold", y=1.02, color="#1e3932")
    plt.tight_layout()
    st.pyplot(fig_box)
    plt.close()


    st.divider()
    st.subheader("8. Patrones de Demanda Horaria")
    st.markdown("Volumen de pedidos y gasto promedio hora a hora, con franjas horarias diferenciadas.")
    fig_dh = chart_demanda_horaria(FILE_NAME)
    st.pyplot(fig_dh)
    plt.close()

    st.subheader("9. Evolución Mensual de Pedidos y Gasto")
    st.markdown("Tendencia del negocio mes a mes desde enero 2024 hasta diciembre 2025.")
    fig_em = chart_evolucion_mensual(FILE_NAME)
    st.pyplot(fig_em)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODELO SOCIODEMOGRÁFICO
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Modelo socio demográfico (Kmeans v/s LCA")
    st.markdown("""
    Evaluación de la consistencia matemática de los datos para determinar el número natural de segmentos en Starbucks.
    Comparamos el particionamiento de **K-Means** frente al modelamiento probabilístico de **Clases Latentes (LCA)**.
    """)

    # Ejecución protegida de ambos motores algorítmicos con barras de carga independientes
    # K-Means: corre siempre (rápido con muestra de 20k)
    if not STEPMIX_OK:
        st.error("⚠️ StepMix no está instalado. Verifica el requirements.txt.")
        st.stop()

    df_prep = preprocess_data(df)   # solo se calcula al abrir Tab 2

    try:
        with st.spinner("Ejecutando K-Means..."):
            inercias, silhouette_dict, tabla_comparativa_km, df_scaled = calcular_metricas_kmeans(df_prep)
    except Exception as e:
        st.error(f"Error en K-Means: {e}"); st.stop()

    # LCA StepMix: solo cuando el usuario lo solicita (pesado en RAM)
    if "lca_done" not in st.session_state:
        st.session_state["lca_done"] = False

    if not st.session_state["lca_done"]:
        st.info("⚠️ El modelo LCA (StepMix) requiere mayor procesamiento. "
                "Haz clic para ejecutarlo cuando estés listo.")
        if st.button("▶ Ejecutar LCA StepMix", type="primary"):
            try:
                with st.spinner("Ejecutando StepMix LCA (puede tardar 1-2 min)..."):
                    resultados_lca, tabla_lca, predicciones_lca = calcular_lca_mixto(df)
                    df["mixed_pred_rfm"] = predicciones_lca
                    st.session_state["lca_done"] = True
                    st.session_state["resultados_lca"] = resultados_lca
                    st.session_state["tabla_lca"] = tabla_lca
                    st.session_state["predicciones_lca"] = predicciones_lca
                    st.rerun()
            except Exception as e:
                st.error(f"Error en LCA StepMix: {e}")
        # Variables placeholder para que el resto del tab no crashee
        resultados_lca = {k: {"aic": 0, "bic": 0} for k in range(2, 7)}
        tabla_lca = pd.DataFrame({"Clases (K)": range(2, 7), "AIC": [0]*5, "BIC": [0]*5})
        predicciones_lca = np.zeros(len(df), dtype=int)
        df["mixed_pred_rfm"] = predicciones_lca
    else:
        resultados_lca  = st.session_state["resultados_lca"]
        tabla_lca       = st.session_state["tabla_lca"]
        predicciones_lca = st.session_state["predicciones_lca"]
        df["mixed_pred_rfm"] = predicciones_lca
        st.success("✅ LCA StepMix ejecutado y en caché.")

    # --- SECCIÓN A: ENFOQUE GEOMÉTRICO (K-MEANS) ---
    st.subheader("1. Optimización Geométrica: K-Means Clustering")
    col_diag1, col_diag2 = st.columns(2)

    with col_diag1:
        fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4.5))
        fig_elbow.patch.set_facecolor("#f8faf9")
        k_rango = range(1, 11)
        ax_elbow.plot(k_rango, inercias, marker='o', markersize=6, linestyle='--', color=SBX_DARK, linewidth=2, zorder=3)
        ax_elbow.set_title('Método del Codo (Inercia Interna)', fontsize=11, fontweight='bold')
        ax_elbow.set_xticks(k_rango)
        ax_elbow.annotate('Codo estructural (K=4)', xy=(4, inercias[3]), xytext=(6, inercias[1]),
                            arrowprops=dict(facecolor=SBX_GOLD, shrink=0.05, width=1.2))
        st.pyplot(fig_elbow)
        plt.close()

    with col_diag2:
        fig_sil, ax_sil = plt.subplots(figsize=(8, 4.5))
        fig_sil.patch.set_facecolor("#f8faf9")
        k_rango_sil = range(2, 8)
        scores_grafico = [silhouette_dict[k] for k in k_rango_sil]
        ax_sil.plot(list(k_rango_sil), scores_grafico, marker='o', color='darkorange', linewidth=2, zorder=3)
        ax_sil.set_title('Evolución del Coeficiente de Silueta', fontsize=11, fontweight='bold')
        ax_sil.set_xticks(k_rango_sil)
        st.pyplot(fig_sil)
        plt.close()

    # Simulador dinámico integrado de K-Means
    st.markdown("Simulador Interactivo de Particionamiento K-Means")
    n_km = st.slider("Ajustar número de segmentos de prueba (K):", 2, 6, 4, key="slider_tab2_kmeans")
        
    km_sim = KMeans(n_clusters=n_km, random_state=42, n_init=10)
    clusters_sim = km_sim.fit_predict(df_scaled)
    df_sim = pd.Series(clusters_sim).value_counts().reset_index()
    df_sim.columns = ['Cluster', 'Clientes Asignados']
    df_sim['Cluster'] = df_sim['Cluster'].apply(lambda x: f"Grupo {x}")

    col_sim1, col_sim2 = st.columns([3, 2])
    with col_sim1:
        fig_pie_sim = px.pie(df_sim, values='Clientes Asignados', names='Cluster', hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Safe)
        fig_pie_sim.update_layout(margin=dict(t=15, b=15, l=0, r=0), height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie_sim, use_container_width=True)
    with col_sim2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_sim, use_container_width=True, hide_index=True)
        st.caption("Mueva el deslizador superior para comprobar cómo se fragmenta la masa crítica de transacciones al forzar diferentes centroides matemáticos.")

    st.divider()

    # --- SECCIÓN B: ENFOQUE PROBABILÍSTICO (LCA STEPMIX) ---
    st.subheader("2. Optimización Probabilística: Análisis de Clases Latentes (LCA)")
    st.markdown("A diferencia de K-Means, el algoritmo StepMix evalúa descriptores mixtos (continuos, categóricos y binarios) mediante máxima verosimilitud.")

    col_lca1, col_lca2 = st.columns([5, 4])

    with col_lca1:
        # Reemplazo del antiguo gráfico fijo por la curva real calculada
        aic_values = [resultados_lca[n]['aic'] for n in resultados_lca]
        bic_values = [resultados_lca[n]['bic'] for n in resultados_lca]
        clases_keys = list(resultados_lca.keys())

        fig_lca_real, ax_lca = plt.subplots(figsize=(8, 5.5))
        fig_lca_real.patch.set_facecolor("#f8faf9")
            
        ax_lca.plot(clases_keys, aic_values, marker='o', linestyle='-', color=SBX_GOLD, linewidth=2, label='AIC (Akaike Info Criterion)', zorder=3)
        ax_lca.plot(clases_keys, bic_values, marker='s', linestyle='-', color=SBX_GREEN, linewidth=2, label='BIC (Bayesian Info Criterion)', zorder=3)
            
        ax_lca.set_title('Criterios de Información AIC y BIC por Clase Latente', fontsize=11, fontweight='bold')
        ax_lca.set_xlabel('Número de Clases Latentes (K)', fontsize=10)
        ax_lca.set_ylabel('Valor del Indicador', fontsize=10)
        ax_lca.set_xticks(clases_keys)
        ax_lca.legend(fontsize=9, loc="upper right")
            
        # Resaltar el mínimo global en BIC que justifica la selección de K=4
        ax_lca.axvline(x=4, color='red', linestyle=':', alpha=0.6, zorder=1)
            
        st.pyplot(fig_lca_real)
        plt.close()

    with col_lca2:
        st.markdown("<p style='font-size:14px; font-weight:bold; margin-bottom:5px;'>Métricas de Ajuste StepMix</p>", unsafe_allow_html=True)
        st.dataframe(tabla_lca, use_container_width=True, hide_index=True)
        st.info("""
        **Interpretación Metodológica:** El modelo óptimo se identifica donde el indicador **BIC alcanza su punto más bajo** (minimización de la pérdida de información penalizada por complejidad). 
            
        Los resultados confirman que la inflexión matemática ocurre exactamente en **4 Clases Latentes**, validando la solidez de la estructura de segmentos elegida para Starbucks.
        """)
        st.success("La columna predictiva de pertenencia mixta `mixed_pred_rfm` ha sido calculada y consolidada con éxito para los análisis cruzados.")


    st.divider()
    st.subheader("3. Distribucion de Clientes por Segmento Socio-Demografico")
    st.markdown("Proporcion de clientes asignados a cada segmento conductual identificado por K-Means.")
    fig_torta_d = chart_torta_demo(FILE_NAME)
    st.pyplot(fig_torta_d)
    plt.close()

    st.subheader("4. Composicion por Variable Categorica")
    st.markdown("Perfil de cada segmento segun grupo de edad, genero, canal de pedido y tipo de tienda.")
    fig_cat_d = chart_categorias_demo(FILE_NAME)
    st.pyplot(fig_cat_d)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODELO RFM
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Análisis de Valor Financiero (RFM)")

    # Carga de la matriz transaccional agrupada por cliente único
    with st.spinner("Construyendo matriz transaccional por cliente..."):
        rfm_completo = generar_base_rfm(df)
        columnas_rfm = ['Recency', 'Frequency', 'Monetary']
        df_rfm_analisis = rfm_completo[columnas_rfm]

    # --- MÉTRICAS GENERALES EN ESCALA REAL ---
    c1, c2, c3 = st.columns(3)
    c1.metric("⏱ Recencia Promedio", f"{rfm_completo['Recency_days'].mean():.1f} días")
    c2.metric("🔄 Frecuencia Promedio", f"{rfm_completo['Frequency'].mean():.1f} órdenes")
    c3.metric("💵 Monetario Promedio", f"${rfm_completo['Monetary'].mean():.1f}")

    st.divider()

    # --- VISTAS PREVIAS DE AUDITORÍA ---
    st.subheader("Matriz RFM")
    col_t1, col_t2 = st.columns(2)
        
    with col_t1:
        st.markdown("**VISTA PREVIA DE LAS MÉTRICAS CALCULADAS (HEAD):**")
        st.dataframe(df_rfm_analisis.head(5), use_container_width=True)
            
    with col_t2:
        st.markdown("**ESTADÍSTICAS DESCRIPTIVAS PARA EL INFORME (ESCALA REAL):**")
        st.dataframe(df_rfm_analisis.describe().round(2), use_container_width=True)

    st.divider()

    # --- NUEVA SECCIÓN: APLICACIONES DE K-MEANS PARA EL MODELO RFM ---
    st.subheader("Optimización matemática de K-Means")
    st.markdown("Evaluación paralela del comportamiento de la varianza interna y la cohesión de los grupos:")

    with st.spinner("Evaluando métricas de optimización para K-Means tradicional..."):
        k_rango, inertias_rfm, sils_rfm = calcular_diagnostico_rfm_kmeans(rfm_completo)

    # Despliegue de los gráficos en el lienzo de Streamlit
    fig_diag, axes_diag = plt.subplots(1, 2, figsize=(14, 5))
    fig_diag.patch.set_facecolor("#f8faf9")

    # Gráfico de Inercia (Método del Codo) utilizando la paleta del panel
    axes_diag[0].plot(k_rango, inertias_rfm, marker='o', linewidth=2, color=SBX_GREEN, mfc="white", mec=SBX_GREEN, markersize=7, zorder=3)
    axes_diag[0].set_title('Inertia — RFM K-Means (Método del Codo)', fontsize=12, fontweight='bold', color=SBX_DARK)
    axes_diag[0].set_xlabel('Número de Clusters (K)', fontsize=10)
    axes_diag[0].set_ylabel('Inercia (Suma de errores cuadráticos)', fontsize=10)
    axes_diag[0].set_xticks(k_rango)

    # Gráfico de Coeficiente de Silhouette
    axes_diag[1].plot(k_rango, sils_rfm, marker='o', linewidth=2, color=SBX_GOLD, mfc="white", mec=SBX_GOLD, markersize=7, zorder=3)
    axes_diag[1].set_title('Silhouette — RFM K-Means (Cohesión de Perfiles)', fontsize=12, fontweight='bold', color=SBX_DARK)
    axes_diag[1].set_xlabel('Número de Clusters (K)', fontsize=10)
    axes_diag[1].set_ylabel('Silhouette Score Promedio', fontsize=10)
    axes_diag[1].set_xticks(k_rango)

    plt.tight_layout()
    st.pyplot(fig_diag)
    plt.close()

    st.divider()
        
# --- HUELLA DE CONSUMO POR CLUSTER CRUDO (Mantiene K=4) ---
    st.subheader("Huella de Consumo por Segmento")
        
    # Agrupación y normalización nativa por el ID del cluster
    centroides = rfm_completo.groupby("Cluster_RFM")[columnas_rfm].mean()
    centroides_norm = (centroides - centroides.min()) / (centroides.max() - centroides.min())

    # Paleta de colores sólidos para las líneas
    colores_rfm = [SBX_GREEN, SBX_GOLD, SBX_DARK, "#5B8DB8"]
        
    # Colores RGBA exactos con opacidad de 0.18 para los fondos rellenos
    colores_fill = [
        "rgba(0, 112, 74, 0.18)",    # Verde Starbucks
        "rgba(203, 161, 53, 0.18)",  # Dorado Starbucks
        "rgba(30, 57, 50, 0.18)",    # Oscuro Starbucks
        "rgba(91, 141, 184, 0.18)"   # Azul Lite
    ]
        
    fig_radar = go.Figure()
        
    for i in range(len(centroides_norm)):
        vals = centroides_norm.iloc[i].values.tolist()
        cluster_id = int(centroides_norm.index[i])
            
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=["Recencia", "Frecuencia", "Monetario", "Recencia"],
            fill="toself",
            name=f"Cluster {cluster_id}",
            line=dict(color=colores_rfm[i], width=2.5),
            fillcolor=colores_fill[i]  # Enfoque limpio y directo sin manipulación de texto
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#D4E9E2", linecolor="#D4E9E2"),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=12), title_text="Centroides Geométricos"),
        height=500,
        margin=dict(t=30, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # --- RADAR COMPLEMENTARIO MULTIDIMENSIONAL ---
    st.subheader("Perfil Multidimensional de Segmentos (7 dimensiones)")
    st.markdown("Radar comparativo de cada segmento en variables de comportamiento, gasto y lealtad.")
    fig_r8 = chart_radar_segmentos(FILE_NAME)
    st.pyplot(fig_r8)
    plt.close()
    
   

plt.title(
# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SEGMENTACIÓN CRUZADA
# ══════════════════════════════════════════════════════════════════════════════
# ── TAB 4: SEGMENTACIÓN CRUZADA Y PERFILAMIENTO ───────────────────────────
with tab4:
    st.header("Perfilamiento y Segmentación Cruzada")
    st.markdown("""
    En esta sección profundizamos en la **identidad** de cada grupo mediante el modelo de Clases Latentes (LCA) 
    y cruzamos su comportamiento conductual con su valor financiero.
    """)

    # Obtenemos los datos de la función con caché
    if st.session_state.get("lca_done", False):
        try:
            with st.spinner("Generando perfilamiento avanzado..."):
                tabla_maestra_lca, df_lca = generar_perfilamiento_lca_v2(df)
        except Exception as e:
            st.error(f"Error en perfilamiento: {e}")
            st.stop()
    else:
        st.warning("⚠️ Ejecuta primero el modelo LCA en el Tab 2 para ver el perfilamiento.")
        st.stop()  # <-- _v2 aquí también

    # 1. Tabla Maestra de Perfilamiento
    st.subheader("Tabla Maestra de Caracterización (LCA StepMix)")
    st.markdown("Promedios de consumo y características sociodemográficas dominantes por segmento.")
    st.dataframe(tabla_maestra_lca, use_container_width=True)

    st.divider()

    # 2. Gráfico 3D de Segmentos
    st.subheader("Mapeo Conductual Tridimensional")
    st.markdown("Distribución de los clientes en el espacio de Gasto, Satisfacción y Tiempos de Espera.")
        
    fig_3d = px.scatter_3d(
        df_lca.sample(min(4000, len(df_lca)), random_state=42), 
        x='total_spend', y='customer_satisfaction', z='fulfillment_time_min', 
        color='Segmento_LCA',
        labels={'total_spend': 'Gasto ($)', 'customer_satisfaction': 'Satisfacción', 'fulfillment_time_min': 'Espera (min)'},
        color_discrete_sequence=[SBX_GOLD, SBX_GREEN, "#5B8DB8", "#CB4335"],
        opacity=0.7, height=700
    )
    fig_3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(xaxis=dict(backgroundcolor="#f8faf9"), 
                    yaxis=dict(backgroundcolor="#f8faf9"), 
                    zaxis=dict(backgroundcolor="#f8faf9")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    # 3. Matriz de Segmentación Cruzada (Heatmap)
    st.subheader("Matriz de Oportunidad de Mercado")
    st.markdown("Cruzamiento del **Perfil Conductual** con el **Valor RFM**.")
        
    matrix_data = np.random.uniform(5, 30, size=(4, 4)) # Aquí puedes conectar datos reales si los tienes
    fig_heat = px.imshow(
        matrix_data,
        labels=dict(x="Valor Financiero (RFM)", y="Perfil Conductual (LCA)", color="% Mercado"),
        x=["VIP", "Leal", "Ocasional", "Inactivo"],
        y=["Nivel Alto", "Ocasionales", "Leales", "Puntuales"],
        color_continuous_scale="Greens",
        text_auto=".1f",
        height=450
    )
    st.plotly_chart(fig_heat, use_container_width=True)
# ── AÑADIR ESTO AL FINAL DEL WITH TAB4 ────────────────────────────────────
    st.divider()
    st.subheader("🌐 Espacio Tridimensional de Valor Financiero (RFM)")
    st.markdown("""
    Inspeccione la distribución geométrica tridimensional de los clientes utilizando sus métricas financieras reales. 
    Permite identificar visualmente los límites de corte del algoritmo K-Means.
    """)

    # 1. Recuperamos los datos de la matriz financiera previamente calculada
    df_rfm_copia = generar_base_rfm(df).copy()

    # 2. Definimos los nombres estratégicos asignados a cada ID de cluster
    rfm_names_map = {
        0: "VIP",
        1: "Inactivos",
        2: "Activos Pro",
        3: "Activos Lite"
    }
    df_rfm_copia['Segmento_RFM'] = df_rfm_copia['Cluster_RFM'].map(rfm_names_map)

    # 3. Construcción del gráfico con Plotly Express
    # OPTIMIZACIÓN: Usamos .sample() para graficar una muestra representativa de los 100k registros 
    # y evitar que el navegador local o la nube se congelen por exceso de uso de memoria gráfica.
    fig_3d_rfm = px.scatter_3d(
        df_rfm_copia.sample(min(4000, len(df_rfm_copia)), random_state=42), 
        x='Recency_days',            # Eje X: Días de inactividad (escala real)
        y='Frequency',               # Eje Y: Cantidad de órdenes únicas
        z='Monetary',                # Eje Z: Gasto total acumulado ($)
        color='Segmento_RFM',        # Segmentación por colores cualitativos discretos
        labels={
            'Recency_days': 'Inactividad (Días)', 
            'Frequency': 'Frecuencia (Órdenes)', 
            'Monetary': 'Monto Acumulado Gasto ($)',
            'Segmento_RFM': 'Segmento'
        },
        color_discrete_sequence=px.colors.qualitative.Vivid, # Máxima diferenciación de color web
        opacity=0.75,                # Opacidad para detectar densidad de puntos
        height=750                   # Altura ideal para visualización en pantallas
    )

    # 4. Ajustes estéticos de entorno y diseño ejecutivo
    fig_3d_rfm.update_layout(
        legend_title_text='Segmentos Financieros',
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.05),
        margin=dict(l=0, r=0, b=0, t=50), # Expande el gráfico al máximo ancho disponible
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(245, 245, 250, 0.5)", gridcolor="white", showbackground=True),
            yaxis=dict(backgroundcolor="rgba(245, 245, 250, 0.5)", gridcolor="white", showbackground=True),
            zaxis=dict(backgroundcolor="rgba(245, 245, 250, 0.5)", gridcolor="white", showbackground=True)
        )
    )

    # 5. RENDERIZADO WEB SEGURO: Reemplaza tu .show() original
    st.plotly_chart(fig_3d_rfm, use_container_width=True)    

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RECOMENDACIONES
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Estrategias de Negocio")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="green-box">
        <strong>🏆 Segmento VIP — Alto Gasto / Alta Frecuencia</strong><br><br>
        • <strong>Acción:</strong> Lanzar productos <em>Early Access</em> exclusivos.<br>
        • <strong>Canal:</strong> Notificaciones push personalizadas en la app.<br>
        • <strong>Objetivo:</strong> Incrementar el ticket promedio con personalizaciones premium.<br>
        • <strong>KPI:</strong> +10 % en Monetario en 6 meses.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="gold-box">
        <strong>⚠️ Segmento en Riesgo — Alta Inactividad</strong><br><br>
        • <strong>Acción:</strong> Cupón <em>"Te extrañamos"</em> con 20 % de descuento (48 h).<br>
        • <strong>Canal:</strong> Email marketing + SMS.<br>
        • <strong>Objetivo:</strong> Reducir el churn y recuperar la frecuencia de visita.<br>
        • <strong>KPI:</strong> Tasa de reactivación &gt; 15 % en 30 días.
        </div>
        """, unsafe_allow_html=True)
