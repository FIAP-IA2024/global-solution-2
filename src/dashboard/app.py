import streamlit as st
import sys
import os

# Configurar o caminho para importações
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuração da página principal
st.set_page_config(
    page_title="Sistema de Monitoramento de Desastres",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Importar páginas modularizadas
import dashboard as dashboard

# Sidebar principal
with st.sidebar:
    st.title("🚨 Monitor de Desastres")
    st.write("FIAP Global Solution 2025.1")

    st.markdown("---")
    st.caption(" 2025 FIAP Global Solution")


# Executar a função show()
if __name__ == "__main__":
    dashboard.show()
