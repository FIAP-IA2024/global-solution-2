import streamlit as st
import sys
import os

# Configurar o caminho para importações
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar páginas modularizadas
import pages.dashboard as dashboard
import pages.alerts as alerts
import pages.devices as devices
import pages.mapview as mapview
import pages.analytics as analytics

# Configuração da página
st.set_page_config(
    page_title="Sistema de Monitoramento de Desastres",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar e navegação
with st.sidebar:
    st.title("🚨 Monitor de Desastres")
    st.write("FIAP Global Solution 2025.1")
    
    st.markdown("---")
    
    # Menu
    page = st.radio("Navegação", ["Dashboard", "Alerts", "Devices", "Maps", "Analytics"])
    
    st.markdown("---")
    st.caption("© 2025 FIAP Global Solution")

# Conteúdo principal
if page == "Dashboard":
    dashboard.show()
elif page == "Alerts":
    alerts.show()
elif page == "Devices":
    devices.show()
elif page == "Maps":
    mapview.show()
elif page == "Analytics":
    analytics.show()

