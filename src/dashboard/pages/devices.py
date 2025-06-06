import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Adicionar o diretório pai ao sys.path para poder importar os módulos utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_devices, load_sensor_data

# Configuração da página
st.set_page_config(
    page_title="Dispositivos - Sistema de Monitoramento de Desastres",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar para navegação
with st.sidebar:
    st.title("🚨 Monitor de Desastres")
    st.write("FIAP Global Solution 2025.1")

    st.markdown("---")
    st.caption("© 2025 FIAP Global Solution")


def show():
    st.title("Dispositivos")
    st.write(
        "Monitoramento e gerenciamento de dispositivos IoT para detecção de desastres."
    )

    # Carregar dispositivos
    devices = load_devices()

    if devices is not None and not devices.empty:
        # Resumo de dispositivos
        st.subheader("Resumo de Dispositivos")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total de Dispositivos", len(devices))

        with col2:
            if "status" in devices.columns:
                online = len(devices[devices["status"] == "online"])
                st.metric(
                    "Dispositivos Online", online, f"{online/len(devices)*100:.1f}%"
                )

        with col3:
            if "status" in devices.columns and "battery" in devices.columns:
                # Dispositivos com bateria baixa (menos de 20%)
                low_battery = len(
                    devices[(devices["status"] == "online") & (devices["battery"] < 20)]
                )
                st.metric("Bateria Baixa", low_battery)

        # Mostrar dispositivos em cards
        st.subheader("Dispositivos")

        # Grid de 3 dispositivos por linha
        cols = st.columns(3)
        col_idx = 0

        for idx, device in devices.iterrows():
            # Determinar cor do status
            status_color = (
                "#4CAF50" if device.get("status", "") == "online" else "#F44336"
            )

            # Determinar ícone de bateria
            battery = device.get("battery", 0)
            if battery > 75:
                battery_icon = "🔋"
            elif battery > 50:
                battery_icon = "🔋"
            elif battery > 25:
                battery_icon = "🪫"
            else:
                battery_icon = "🪫"

            with cols[col_idx]:
                st.markdown(
                    f"""
                <div style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="margin-top: 0;">{device.get('name', device.get('device_id', 'Desconhecido'))}</h3>
                    
                    <p><strong>ID:</strong> {device.get('device_id', 'N/A')}</p>
                    <p><strong>Localização:</strong> {device.get('location', 'N/A')}</p>
                    <p><strong>Status:</strong> <span style="color: {status_color}; font-weight: bold;">{device.get('status', 'N/A')}</span></p>
                    <p><strong>Bateria:</strong> {battery}% {battery_icon}</p>
                    <p><strong>Última atualização:</strong> {device.get('last_update', 'N/A')}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Incrementar índice de coluna, resetar se necessário
                col_idx = (col_idx + 1) % 3

        # Opção para mostrar dados brutos
        if st.checkbox("Mostrar dados brutos"):
            st.dataframe(devices)

        # Gráfico de status dos dispositivos se houver dados suficientes
        if "status" in devices.columns and len(devices) >= 2:
            st.subheader("Status dos Dispositivos")
            status_counts = devices["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]

            fig = px.pie(
                status_counts,
                values="count",
                names="status",
                title="Status dos Dispositivos",
                color="status",
                color_discrete_map={
                    "online": "#4CAF50",
                    "offline": "#F44336",
                    "maintenance": "#FFC107",
                    "error": "#9C27B0",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Nenhum dispositivo encontrado. Verifique a conexão com os dispositivos ESP32."
        )


# Executar a função show()
if __name__ == "__main__":
    show()
