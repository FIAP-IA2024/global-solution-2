import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import sys
import os

# Adicionar o diretório pai ao sys.path para poder importar os módulos utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_alerts

# Configuração da página
st.set_page_config(
    page_title="Alertas - Sistema de Monitoramento de Desastres",
    page_icon="🚨",
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
    st.title("Alertas")
    st.write(
        "Sistema de monitoramento e gerenciamento de alertas para desastres naturais."
    )

    # Carregar dados de alertas
    alerts = load_alerts()

    # Mostrar estatísticas de alertas
    if alerts is not None and not alerts.empty:
        st.subheader("Estatísticas")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total de Alertas", len(alerts))

            # Alertas críticos
            if "severity" in alerts.columns:
                critical = len(alerts[alerts["severity"] == "critical"])
                st.metric("Alertas Críticos", critical)

        with col2:
            if "device_id" in alerts.columns:
                # Dispositivos com alertas
                devices_with_alerts = len(alerts["device_id"].unique())
                st.metric("Dispositivos com Alertas", devices_with_alerts)

        # Verificar as colunas disponíveis para filtros
        if "severity" in alerts.columns:
            # Filtro por severidade
            severity_filter = st.multiselect(
                "Filtrar por severidade:",
                options=sorted(alerts["severity"].unique()),
                default=sorted(alerts["severity"].unique()),
            )

            # Aplicar filtro
            filtered_alerts = alerts[alerts["severity"].isin(severity_filter)]

            if "type" in filtered_alerts.columns:
                alert_types = st.multiselect(
                    "Filtrar por Tipo:",
                    options=sorted(filtered_alerts["type"].unique()),
                    default=sorted(filtered_alerts["type"].unique()),
                )
                filtered_alerts = filtered_alerts[
                    filtered_alerts["type"].isin(alert_types)
                ]
        else:
            filtered_alerts = alerts

        # Exibir alertas em cards
        st.subheader("Alertas Recentes")

        # Ordenar do mais recente para o mais antigo
        if "timestamp" in filtered_alerts.columns:
            filtered_alerts = filtered_alerts.sort_values("timestamp", ascending=False)

        for idx, alert in filtered_alerts.iterrows():
            # Definir cor do card com base na severidade
            if "severity" in alert:
                if alert["severity"].lower() == "critical":
                    card_color = "#FF5252"
                elif alert["severity"].lower() == "high":
                    card_color = "#FFAB40"
                elif alert["severity"].lower() == "medium":
                    card_color = "#FFEB3B"
                else:
                    card_color = "#81C784"
            else:
                card_color = "#E0E0E0"

            # Montar conteúdo do card
            title = alert["type"] if "type" in alert else "Alerta"
            message = (
                alert["message"] if "message" in alert else "Sem descrição disponível"
            )
            timestamp = alert["timestamp"] if "timestamp" in alert else ""
            device = alert["device_id"] if "device_id" in alert else ""

            # Criar card com HTML/CSS
            st.markdown(
                f"""
            <div style="padding: 15px; border-radius: 5px; background-color: {card_color}; margin-bottom: 10px;">
                <h3>{title}</h3>
                <p>{message}</p>
                <p><strong>Dispositivo:</strong> {device}</p>
                <p><small>Data: {timestamp}</small></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Gráfico de contagem de alertas por tipo
        if "type" in filtered_alerts.columns and not filtered_alerts.empty:
            alert_counts = filtered_alerts["type"].value_counts().reset_index()
            alert_counts.columns = ["type", "count"]

            fig = px.bar(
                alert_counts,
                x="type",
                y="count",
                title="Alertas por Tipo",
                labels={"count": "Quantidade", "type": "Tipo de Alerta"},
                color="type",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Opção para mostrar dados brutos
        if st.checkbox("Mostrar dados brutos"):
            st.dataframe(filtered_alerts)
    else:
        st.info("Não há alertas para exibir.")


# Executar a função show()
if __name__ == "__main__":
    show()
