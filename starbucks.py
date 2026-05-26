# VERSIÓN DE DIAGNÓSTICO — reemplazar por la versión completa una vez funcione
import streamlit as st
import os

st.set_page_config(page_title="Starbucks Test", page_icon="☕", layout="wide")

st.title("☕ Starbucks Analytics — Diagnóstico")
st.markdown("---")

# Test 1: Archivo de datos
FILE_NAME = "s_order.csv"
if os.path.exists(FILE_NAME):
    st.success(f"✅ s_order.csv encontrado")
else:
    st.error(f"❌ s_order.csv NO encontrado")

# Test 2: Imports pesados
col1, col2, col3 = st.columns(3)

with col1:
    try:
        import pandas as pd
        st.success(f"✅ pandas {pd.__version__}")
    except Exception as e:
        st.error(f"❌ pandas: {e}")

    try:
        import numpy as np
        st.success(f"✅ numpy {np.__version__}")
    except Exception as e:
        st.error(f"❌ numpy: {e}")

    try:
        import matplotlib
        st.success(f"✅ matplotlib {matplotlib.__version__}")
    except Exception as e:
        st.error(f"❌ matplotlib: {e}")

with col2:
    try:
        import sklearn
        st.success(f"✅ scikit-learn {sklearn.__version__}")
    except Exception as e:
        st.error(f"❌ scikit-learn: {e}")

    try:
        import plotly
        st.success(f"✅ plotly {plotly.__version__}")
    except Exception as e:
        st.error(f"❌ plotly: {e}")

    try:
        import seaborn as sns
        st.success(f"✅ seaborn {sns.__version__}")
    except Exception as e:
        st.error(f"❌ seaborn: {e}")

with col3:
    try:
        from stepmix.stepmix import StepMix
        from stepmix.utils import get_mixed_descriptor
        st.success("✅ stepmix OK")
    except Exception as e:
        st.error(f"❌ stepmix: {e}")

    try:
        import holidays
        st.success(f"✅ holidays {holidays.__version__}")
    except Exception as e:
        st.error(f"❌ holidays: {e}")

    try:
        import statsmodels
        st.success(f"✅ statsmodels {statsmodels.__version__}")
    except Exception as e:
        st.error(f"❌ statsmodels: {e}")

# Test 3: Carga de datos
st.markdown("---")
if os.path.exists(FILE_NAME) and st.button("🔍 Probar carga de datos"):
    import pandas as pd
    with st.spinner("Cargando..."):
        df = pd.read_csv(FILE_NAME)
        st.success(f"✅ Datos cargados: {len(df):,} filas × {df.shape[1]} columnas")
        st.dataframe(df.head(3))

st.info("Si ves esto, la app cargó correctamente. "
        "Reemplaza este archivo por la versión completa.")
