import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

# Adicionar o diretório pai ao sys.path para poder importar os módulos utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_devices, load_sensor_data

def show():
    st.title("Dispositivos e Sensores")
    st.write("Monitoramento de dispositivos ESP32 e sensores ambientais distribuídos.")
    
    # Carregar dispositivos
    devices = load_devices()
    sensor_data = load_sensor_data()
    
    # Estatu00edsticas de dispositivos
    if len(devices) > 0:
        # Contagem por status
        online_count = len(devices[devices["status"] == "online"])
        offline_count = len(devices[devices["status"] == "offline"])
        warning_count = len(devices[devices["status"] == "warning"])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Online", online_count, delta=None, delta_color="normal")
        col2.metric("Offline", offline_count, delta=None, delta_color="inverse")
        col3.metric("Alerta", warning_count, delta=None, delta_color="off")
    
    # Adicionar novo dispositivo
    with st.expander("Adicionar Novo Dispositivo"):
        col1, col2 = st.columns(2)
        with col1:
            device_id = st.text_input("ID do Dispositivo", placeholder="ESP32_xx")
            device_name = st.text_input("Nome do Dispositivo", placeholder="Sensor São Paulo")
            device_location = st.text_input("Localização", placeholder="São Paulo, SP")
        
        with col2:
            device_lat = st.number_input("Latitude", value=-23.5505, format="%f")
            device_lon = st.number_input("Longitude", value=-46.6333, format="%f")
            device_status = st.selectbox("Status", ["online", "offline", "warning"])
        
        if st.button("Adicionar Dispositivo"):
            st.success("Dispositivo adicionado com sucesso! (Simulação)")
    
    # Mapa de dispositivos
    st.subheader("Mapa de Dispositivos")
    
    if len(devices) > 0 and 'lat' in devices.columns and 'lon' in devices.columns:
        # Preparar dados para o mapa
        map_data = pd.DataFrame({
            'lat': devices['lat'],
            'lon': devices['lon'],
        })
        
        st.map(map_data)
        
        # Legenda para o mapa
        st.write("### Dispositivos no Mapa")
        for _, device in devices.iterrows():
            status_color = "green" if device["status"] == "online" else "red" if device["status"] == "offline" else "orange"
            st.markdown(f"<span style='color: {status_color};'>⬤</span> {device['name']} - {device['location']}", 
                        unsafe_allow_html=True)
    else:
        st.warning("Não há dados de localização disponíveis para exibir no mapa")
    
    # Lista de dispositivos com detalhes
    st.subheader("Lista de Dispositivos")
    
    if len(devices) > 0:
        for _, device in devices.iterrows():
            status_color = "green" if device["status"] == "online" else "red" if device["status"] == "offline" else "orange"
            with st.expander(f"{device['name']} ({device['id']}) - Status: {device['status']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"<strong>ID:</strong> {device['id']}<br>"
                               f"<strong>Nome:</strong> {device['name']}<br>"
                               f"<strong>Localização:</strong> {device['location']}<br>"
                               f"<strong>Status:</strong> <span style='color: {status_color};'>{device['status']}</span>", 
                               unsafe_allow_html=True)
                
                with col2:
                    if 'last_update' in device:
                        st.markdown(f"<strong>Última Atualização:</strong> {device['last_update']}<br>"
                                   f"<strong>Coordenadas:</strong> {device.get('lat', 'N/A')}, {device.get('lon', 'N/A')}", 
                                   unsafe_allow_html=True)
                
                # Se o dispositivo estiver online, mostrar gráfico recente
                if device['status'] == "online":
                    st.write("#### Últimas Leituras")
                    # Filtrar dados do dispositivo (mock - na implementação real usaria device_id)
                    device_data = sensor_data.iloc[-24:] # Últimas 24 horas
                    
                    # Gráfico de temperatura
                    fig = px.line(device_data, x='timestamp', y=['temperature', 'humidity'],
                                  title=f'Leituras Recentes - {device["name"]}',
                                  labels={'value': 'Valor', 'timestamp': 'Tempo', 'variable': 'Sensor'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Botões de ação
                col1, col2, col3 = st.columns([1,1,2])
                with col1:
                    if st.button("Editar", key=f"edit_{device['id']}"):
                        st.info("Função para editar dispositivo")
                with col2:
                    if st.button("Reiniciar", key=f"restart_{device['id']}"):
                        st.success("Comando de reinicialização enviado!")
                with col3:
                    if st.button("Ver Detalhes Completos", key=f"details_{device['id']}"):
                        st.info("Função para ver detalhes completos do dispositivo")
    else:
        st.warning("Nenhum dispositivo encontrado")
        
    # Adicionar seção de configuração de sensores
    st.markdown("---")
    st.subheader("Configuração de Limiares de Alerta")
    
    with st.form("threshold_settings"):
        st.write("Defina os limiares para geração de alertas automáticos:")
        
        col1, col2 = st.columns(2)
        with col1:
            temp_high = st.slider("Temperatura Alta (°C)", 20.0, 50.0, 35.0)
            humid_high = st.slider("Umidade Alta (%)", 50.0, 100.0, 80.0)
            pressure_low = st.slider("Pressão Baixa (hPa)", 950.0, 1020.0, 1000.0)
        
        with col2:
            water_level_high = st.slider("Nível de Água Alto (cm)", 10.0, 100.0, 50.0)
            vib_high = st.slider("Vibração Alta (Hz)", 50.0, 500.0, 200.0)
            rain_high = st.slider("Nível de Chuva Alto (mm)", 5.0, 50.0, 20.0)
        
        submitted = st.form_submit_button("Salvar Configurações")
        if submitted:
            st.success("Configurações de limiares atualizadas com sucesso!")
