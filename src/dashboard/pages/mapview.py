import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import sys
import os

# Adicionar o diretório pai ao sys.path para poder importar os módulos utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_devices, load_alerts

# Configuração da página
st.set_page_config(
    page_title="Mapas - Sistema de Monitoramento de Desastres",
    page_icon="🗺️",
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
    st.title("Mapa de Monitoramento")
    st.write("Visualização geográfica de dispositivos, alertas e zonas de risco.")

    # Carregar dados
    devices = load_devices()
    alerts = load_alerts()

    # Filtros para o mapa
    col1, col2 = st.columns(2)
    with col1:
        show_devices = st.checkbox("Mostrar Dispositivos", value=True)
        show_alerts = st.checkbox("Mostrar Alertas", value=True)

    with col2:
        device_status = st.multiselect(
            "Status dos Dispositivos",
            options=["online", "offline", "warning"],
            default=["online", "warning"],
        )

    # Filtrar dispositivos com base no status selecionado
    if show_devices and len(devices) > 0 and "status" in devices.columns:
        filtered_devices = devices[devices["status"].isin(device_status)]
    else:
        filtered_devices = pd.DataFrame()

    # Usar st.map para mapa básico (mais simples)
    if (
        show_devices
        and len(filtered_devices) > 0
        and "lat" in filtered_devices.columns
        and "lon" in filtered_devices.columns
    ):
        st.subheader("Mapa de Dispositivos")
        map_data = pd.DataFrame(
            {
                "lat": filtered_devices["lat"],
                "lon": filtered_devices["lon"],
            }
        )
        st.map(map_data)

    # Usar pydeck para um mapa mais avançado (opcional se st.map for suficiente)
    if (
        st.checkbox("Mostrar Mapa Avançado", value=False)
        and len(filtered_devices) > 0
    ):
        st.subheader("Mapa Avançado")

        # Preparar dados para o mapa
        if (
            len(filtered_devices) > 0
            and "lat" in filtered_devices.columns
            and "lon" in filtered_devices.columns
        ):
            # Definir cores por status
            filtered_devices["color"] = filtered_devices["status"].apply(
                lambda x: (
                    [0, 255, 0, 200]
                    if x == "online"
                    else [255, 0, 0, 200] if x == "offline" else [255, 165, 0, 200]
                )
            )

            # Definir view inicial (cenário para São Paulo, ajuste conforme necessário)
            view_state = pdk.ViewState(
                latitude=-23.5505,
                longitude=-46.6333,
                zoom=10,
                pitch=50,
            )

            # Definir camada de pontos
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered_devices,
                get_position=["lon", "lat"],
                get_color="color",
                get_radius=1000,  # Tamanho em metros
                pickable=True,
                auto_highlight=True,
            )

            # Renderizar mapa
            st.pydeck_chart(
                pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v10",
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={
                        "html": "<b>Nome:</b> {name}<br><b>Status:</b> {status}<br><b>Localização:</b> {location}",
                        "style": {"backgroundColor": "steelblue", "color": "white"},
                    },
                )
            )

    # Zonas de risco (simuladas)
    st.subheader("Zonas de Risco")

    # Criar dados simulados de zonas de risco
    risk_zones = pd.DataFrame(
        {
            "name": [
                "Zona 1 - Alto Risco",
                "Zona 2 - Médio Risco",
                "Zona 3 - Baixo Risco",
            ],
            "risk_level": ["Alto", "Médio", "Baixo"],
            "lat": [
                -23.5505,
                -23.6505,
                -23.7505,
            ],  # Exemplo - ajuste para coordenadas reais
            "lon": [
                -46.6333,
                -46.7333,
                -46.8333,
            ],  # Exemplo - ajuste para coordenadas reais
            "radius": [5000, 3000, 2000],  # Raio em metros
            "description": [
                "Região com alto risco de inundação devido à proximidade de rios e áreas de baixada",
                "Região com risco médio de deslizamento de terra em épocas de chuva intensa",
                "Região com baixo risco de eventos climáticos extremos, mas com monitoramento preventivo",
            ],
        }
    )

    # Exibir zonas de risco em formato de tabela
    for idx, zone in risk_zones.iterrows():
        risk_color = (
            "red"
            if zone["risk_level"] == "Alto"
            else "orange" if zone["risk_level"] == "Médio" else "blue"
        )
        with st.expander(f"{zone['name']}"):
            st.markdown(
                f"<strong>Nível de Risco:</strong> <span style='color: {risk_color};'>{zone['risk_level']}</span>",
                unsafe_allow_html=True,
            )
            st.write(f"**Descrição:** {zone['description']}")
            st.write(f"**Coordenadas:** {zone['lat']}, {zone['lon']}")
            st.write(f"**Raio da Zona:** {zone['radius']/1000:.1f} km")

            # Botões de ação
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Ver Dispositivos na Zona", key=f"devices_{idx}"):
                    st.info(
                        "Função para listar dispositivos na zona (simulação)"
                    )
            with col2:
                if st.button("Ver Histórico de Alertas", key=f"alerts_{idx}"):
                    st.info(
                        "Função para mostrar histórico de alertas da zona (simulação)"
                    )

    # Seção para reportar novas zonas de risco
    st.markdown("---")
    st.subheader("Reportar Nova Zona de Risco")

    with st.form("report_risk_zone"):
        col1, col2 = st.columns(2)

        with col1:
            zone_name = st.text_input("Nome da Zona")
            risk_level = st.select_slider(
                "Nível de Risco",
                options=["Baixo", "Médio", "Alto"],
                value="Médio",
            )
            description = st.text_area("Descrição")

        with col2:
            zone_lat = st.number_input("Latitude", value=-23.5505, format="%f")
            zone_lon = st.number_input("Longitude", value=-46.6333, format="%f")
            zone_radius = st.slider("Raio (km)", 1.0, 10.0, 3.0, 0.5)

        if st.form_submit_button("Registrar Zona de Risco"):
            if zone_name and description:
                st.success(
                    f"Zona de risco '{zone_name}' registrada com sucesso! (Simulação)"
                )
            else:
                st.error("Por favor, preencha todos os campos obrigatórios")


# Executar a função show()
if __name__ == "__main__":
    show()
