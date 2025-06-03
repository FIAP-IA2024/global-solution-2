import streamlit as st
import pandas as pd
import datetime
import sys
import os

# Adicionar o diretório pai ao sys.path para poder importar os módulos utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_alerts

def show():
    st.title("Alertas")
    st.write("Sistema de monitoramento e gerenciamento de alertas para desastres naturais.")
    
    # Carregar alertas
    alerts = load_alerts()
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    with col1:
        severity_filter = st.multiselect("Severidade", ["high", "medium", "low"], default=["high", "medium", "low"])
    
    with col2:
        if len(alerts) > 0 and 'type' in alerts.columns:
            type_filter = st.multiselect("Tipo", alerts["type"].unique().tolist(), default=alerts["type"].unique().tolist())
        else:
            type_filter = st.multiselect("Tipo", ["high_temperature", "vibration", "water_level", "low_pressure"], 
                                      default=["high_temperature", "vibration", "water_level", "low_pressure"])
    
    with col3:
        date_range = st.date_input("Período", 
                                   [datetime.datetime.now() - datetime.timedelta(days=7), datetime.datetime.now()])
    
    # Estatísticas de alertas
    st.subheader("Estatísticas de Alertas")
    
    if len(alerts) > 0 and 'severity' in alerts.columns:
        high_count = len(alerts[alerts["severity"] == "high"])
        medium_count = len(alerts[alerts["severity"] == "medium"])
        low_count = len(alerts[alerts["severity"] == "low"])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Críticos", high_count, delta=None, delta_color="inverse")
        col2.metric("Médios", medium_count, delta=None, delta_color="inverse")
        col3.metric("Baixos", low_count, delta=None, delta_color="inverse")
        
        # Filtrar dados
        if 'type' in alerts.columns:
            filtered_alerts = alerts[
                (alerts["severity"].isin(severity_filter)) & 
                (alerts["type"].isin(type_filter))
            ]
        else:
            filtered_alerts = alerts[
                (alerts["severity"].isin(severity_filter))
            ]
        
        # Mostrar alertas
        st.subheader("Lista de Alertas")
        
        if len(filtered_alerts) > 0:
            for _, alert in filtered_alerts.iterrows():
                severity_color = "red" if alert["severity"] == "high" else "orange" if alert["severity"] == "medium" else "blue"
                with st.expander(f"{alert['timestamp'].strftime('%Y-%m-%d %H:%M')} - {alert['message']}"):
                    st.markdown(f"<strong>ID:</strong> {alert['id']}<br>"
                               f"<strong>Tipo:</strong> {alert.get('type', 'N/A')}<br>"
                               f"<strong>Severidade:</strong> <span style='color: {severity_color};'>{alert['severity']}</span><br>"
                               f"<strong>Timestamp:</strong> {alert['timestamp']}<br>"
                               f"<strong>Mensagem:</strong> {alert['message']}", unsafe_allow_html=True)
                    
                    # Botões de ação para gerenciamento de alertas
                    col1, col2, col3 = st.columns([1,1,2])
                    with col1:
                        if st.button("Marcar como Resolvido", key=f"resolve_{alert['id']}"):
                            st.success("Alerta marcado como resolvido!")
                    with col2:
                        if st.button("Atribuir", key=f"assign_{alert['id']}"):
                            st.info("Função para atribuir alerta a um responsável")
        else:
            st.info("Nenhum alerta corresponde aos filtros selecionados")
    else:
        st.warning("Não há alertas disponíveis no momento")
        
    # Sistema de criação de alertas manuais
    st.markdown("---")
    st.subheader("Criar Alerta Manual")
    
    with st.form("create_alert_form"):
        col1, col2 = st.columns(2)
        with col1:
            alert_message = st.text_input("Mensagem do Alerta")
            alert_type = st.selectbox("Tipo", ["high_temperature", "vibration", "water_level", "low_pressure", "other"])
            
        with col2:
            alert_severity = st.select_slider("Severidade", options=["low", "medium", "high"], value="medium")
            location = st.text_input("Localização")
        
        submitted = st.form_submit_button("Criar Alerta")
        if submitted:
            if alert_message:
                st.success(f"Alerta '{alert_message}' criado com sucesso!")
            else:
                st.error("Por favor, preencha a mensagem do alerta")
