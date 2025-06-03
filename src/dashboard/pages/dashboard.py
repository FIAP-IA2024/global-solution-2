import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import sys
import os

# Adicionar o diretório pai ao sys.path para poder importar os módulos utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_sensor_data, load_alerts

def show():
    st.title("Dashboard")
    st.write("Real-time monitoring of environmental conditions for disaster prediction.")
    
    # Carregar dados
    data = load_sensor_data()
    alerts = load_alerts()
    
    # Estatísticas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Temperature", f"{data['temperature'].iloc[-1]:.1f}°C", 
                 f"{data['temperature'].iloc[-1] - data['temperature'].iloc[-2]:.1f}°C")
    
    with col2:
        st.metric("Humidity", f"{data['humidity'].iloc[-1]:.1f}%", 
                 f"{data['humidity'].iloc[-1] - data['humidity'].iloc[-2]:.1f}%")
    
    with col3:
        st.metric("Pressure", f"{data['pressure'].iloc[-1]:.1f} hPa", 
                 f"{data['pressure'].iloc[-1] - data['pressure'].iloc[-2]:.1f}")
    
    with col4:
        st.metric("Water Level", f"{data['water_level'].iloc[-1]:.1f} cm", 
                 f"{data['water_level'].iloc[-1] - data['water_level'].iloc[-2]:.1f} cm")
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Over Time")
        fig = px.line(data, x='timestamp', y='temperature', title='Temperature Trend')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Humidity Over Time")
        fig = px.line(data, x='timestamp', y='humidity', title='Humidity Trend')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pressure Over Time")
        fig = px.line(data, x='timestamp', y='pressure', title='Pressure Trend')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Recent Alerts")
        if len(alerts) > 0:
            for _, alert in alerts.iterrows():
                severity_color = "red" if alert["severity"] == "high" else "orange" if alert["severity"] == "medium" else "blue"
                st.markdown(f"<div style='padding: 10px; border-left: 5px solid {severity_color}; margin-bottom: 10px;'>"
                           f"<strong>{alert['timestamp'].strftime('%Y-%m-%d %H:%M')}</strong><br>"
                           f"{alert['message']}<br>"
                           f"<small>Severity: {alert['severity']}</small>"
                           f"</div>", unsafe_allow_html=True)
        else:
            st.write("No recent alerts")
    
    # Visão geral de todos os sensores
    st.subheader("All Sensors Overview")
    
    # Selecionar intervalo de tempo
    time_range = st.select_slider(
        "Time Range",
        options=["Last Hour", "Last 12 Hours", "Last Day", "Last 3 Days", "Last Week"],
        value="Last Day"
    )
    
    # Filtrar dados com base no intervalo selecionado
    if time_range == "Last Hour":
        filtered_data = data[data['timestamp'] > (datetime.datetime.now() - datetime.timedelta(hours=1))]
    elif time_range == "Last 12 Hours":
        filtered_data = data[data['timestamp'] > (datetime.datetime.now() - datetime.timedelta(hours=12))]
    elif time_range == "Last Day":
        filtered_data = data[data['timestamp'] > (datetime.datetime.now() - datetime.timedelta(days=1))]
    elif time_range == "Last 3 Days":
        filtered_data = data[data['timestamp'] > (datetime.datetime.now() - datetime.timedelta(days=3))]
    else:  # Last Week
        filtered_data = data
    
    # Converter dados para formato longo para plotagem
    sensors = ['temperature', 'humidity', 'pressure', 'water_level', 'soil_moisture', 'vibration', 'rain_level']
    long_data = pd.melt(filtered_data, id_vars=['timestamp'], value_vars=sensors, var_name='sensor', value_name='value')
    
    # Plotar gráfico de todos os sensores
    fig = px.line(long_data, x='timestamp', y='value', color='sensor', facet_row='sensor',
                 labels={'value': 'Value', 'timestamp': 'Time'},
                 title=f'All Sensors - {time_range}')
    
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)
