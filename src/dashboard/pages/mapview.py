import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import sys
import os

# Adicionar o diretório pai ao sys.path para poder importar os módulos utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_devices, load_alerts

def show():
    st.title("Mapa de Monitoramento")
    st.write("Visualizau00e7u00e3o geogru00e1fica de dispositivos, alertas e zonas de risco.")
    
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
            default=["online", "warning"]
        )
    
    # Filtrar dispositivos com base no status selecionado
    if show_devices and len(devices) > 0 and 'status' in devices.columns:
        filtered_devices = devices[devices["status"].isin(device_status)]
    else:
        filtered_devices = pd.DataFrame()
    
    # Usar st.map para mapa bu00e1sico (mais simples)
    if show_devices and len(filtered_devices) > 0 and 'lat' in filtered_devices.columns and 'lon' in filtered_devices.columns:
        st.subheader("Mapa de Dispositivos")
        map_data = pd.DataFrame({
            'lat': filtered_devices['lat'],
            'lon': filtered_devices['lon'],
        })
        st.map(map_data)
    
    # Usar pydeck para um mapa mais avanu00e7ado (opcional se st.map for suficiente)
    if st.checkbox("Mostrar Mapa Avanu00e7ado", value=False) and len(filtered_devices) > 0:
        st.subheader("Mapa Avanu00e7ado")
        
        # Preparar dados para o mapa
        if len(filtered_devices) > 0 and 'lat' in filtered_devices.columns and 'lon' in filtered_devices.columns:
            # Definir cores por status
            filtered_devices['color'] = filtered_devices['status'].apply(
                lambda x: [0, 255, 0, 200] if x == 'online' else [255, 0, 0, 200] if x == 'offline' else [255, 165, 0, 200]
            )
            
            # Definir view inicial (cenu00e1rio para Su00e3o Paulo, ajuste conforme necessu00e1rio)
            view_state = pdk.ViewState(
                latitude=-23.5505,
                longitude=-46.6333,
                zoom=10,
                pitch=50,
            )
            
            # Definir camada de pontos
            layer = pdk.Layer(
                'ScatterplotLayer',
                data=filtered_devices,
                get_position=['lon', 'lat'],
                get_color='color',
                get_radius=1000,  # Tamanho em metros
                pickable=True,
                auto_highlight=True,
            )
            
            # Renderizar mapa
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v10',
                initial_view_state=view_state,
                layers=[layer],
                tooltip={
                    'html': '<b>Nome:</b> {name}<br><b>Status:</b> {status}<br><b>Localizau00e7u00e3o:</b> {location}',
                    'style': {
                        'backgroundColor': 'steelblue',
                        'color': 'white'
                    }
                }
            ))
    
    # Zonas de risco (simuladas)
    st.subheader("Zonas de Risco")
    
    # Criar dados simulados de zonas de risco
    risk_zones = pd.DataFrame({
        'name': ['Zona 1 - Alto Risco', 'Zona 2 - Mu00e9dio Risco', 'Zona 3 - Baixo Risco'],
        'risk_level': ['Alto', 'Mu00e9dio', 'Baixo'],
        'lat': [-23.5505, -23.6505, -23.7505],  # Exemplo - ajuste para coordenadas reais
        'lon': [-46.6333, -46.7333, -46.8333],  # Exemplo - ajuste para coordenadas reais
        'radius': [5000, 3000, 2000],  # Raio em metros
        'description': [
            'Regiu00e3o com alto risco de inundau00e7u00e3o devido u00e0 proximidade de rios e u00e1reas de baixada',
            'Regiu00e3o com risco mu00e9dio de deslizamento de terra em u00e9pocas de chuva intensa',
            'Regiu00e3o com baixo risco de eventos climu00e1ticos extremos, mas com monitoramento preventivo'
        ]
    })
    
    # Exibir zonas de risco em formato de tabela
    for idx, zone in risk_zones.iterrows():
        risk_color = "red" if zone["risk_level"] == "Alto" else "orange" if zone["risk_level"] == "Mu00e9dio" else "blue"
        with st.expander(f"{zone['name']}"):
            st.markdown(f"<strong>Nu00edvel de Risco:</strong> <span style='color: {risk_color};'>{zone['risk_level']}</span>", 
                       unsafe_allow_html=True)
            st.write(f"**Descriu00e7u00e3o:** {zone['description']}")
            st.write(f"**Coordenadas:** {zone['lat']}, {zone['lon']}")
            st.write(f"**Raio da Zona:** {zone['radius']/1000:.1f} km")
            
            # Botu00f5es de au00e7u00e3o
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Ver Dispositivos na Zona", key=f"devices_{idx}"):
                    st.info("Funu00e7u00e3o para listar dispositivos na zona (simulau00e7u00e3o)")
            with col2:
                if st.button("Ver Histu00f3rico de Alertas", key=f"alerts_{idx}"):
                    st.info("Funu00e7u00e3o para mostrar histu00f3rico de alertas da zona (simulau00e7u00e3o)")
    
    # Seu00e7u00e3o para reportar novas zonas de risco
    st.markdown("---")
    st.subheader("Reportar Nova Zona de Risco")
    
    with st.form("report_risk_zone"):
        col1, col2 = st.columns(2)
        
        with col1:
            zone_name = st.text_input("Nome da Zona")
            risk_level = st.select_slider("Nu00edvel de Risco", options=["Baixo", "Mu00e9dio", "Alto"], value="Mu00e9dio")
            description = st.text_area("Descriu00e7u00e3o")
        
        with col2:
            zone_lat = st.number_input("Latitude", value=-23.5505, format="%f")
            zone_lon = st.number_input("Longitude", value=-46.6333, format="%f")
            zone_radius = st.slider("Raio (km)", 1.0, 10.0, 3.0, 0.5)
        
        if st.form_submit_button("Registrar Zona de Risco"):
            if zone_name and description:
                st.success(f"Zona de risco '{zone_name}' registrada com sucesso! (Simulau00e7u00e3o)")
            else:
                st.error("Por favor, preencha todos os campos obrigatu00f3rios")
