import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Adicionar o diretório pai ao sys.path para poder importar os módulos utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_disaster_dataset, load_prediction_results
from utils.model_loader import (
    simulate_ml_prediction,
    load_all_available_models,
    get_feature_importance,
)

# Configuração da página
st.set_page_config(
    page_title="Analytics - Sistema de Monitoramento de Desastres",
    page_icon="📈",
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
    st.title("Analytics")
    st.write(
        "Anu00e1lise de dados histu00f3ricos de desastres e prediu00e7u00f5es do modelo de Machine Learning."
    )

    # Carregar dados de desastres histu00f3ricos
    df_disasters = load_disaster_dataset()
    predictions = load_prediction_results()

    # Tabs para separar os diferentes tipos de anu00e1lise
    tab1, tab2, tab3 = st.tabs(
        ["Dados Histu00f3ricos", "Prediu00e7u00f5es", "Modelo de ML"]
    )

    with tab1:
        st.header("Anu00e1lise de Dados Histu00f3ricos")

        if df_disasters is not None and not df_disasters.empty:
            # Resumo estatu00edstico dos dados
            st.subheader("Resumo Estatu00edstico")

            # Verificar as colunas disponu00edveis no dataframe
            if all(
                col in df_disasters.columns
                for col in ["Disaster Type", "Total Deaths", "Total Affected"]
            ):
                # Agrupar por tipo de desastre
                disaster_summary = (
                    df_disasters.groupby("Disaster Type")
                    .agg({"Total Deaths": "sum", "Total Affected": "sum"})
                    .reset_index()
                )

                # Mostrar gru00e1fico de barras por tipo de desastre
                st.write("#### Impacto por Tipo de Desastre")

                impact_metric = st.selectbox(
                    "Selecione o mu00e9trico de impacto",
                    options=["Total Deaths", "Total Affected"],
                )

                fig = px.bar(
                    disaster_summary,
                    x="Disaster Type",
                    y=impact_metric,
                    title=f"{impact_metric} por Tipo de Desastre",
                    color="Disaster Type",
                    labels={
                        "value": impact_metric,
                        "Disaster Type": "Tipo de Desastre",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

                # Anu00e1lise temporal
                if "Year" in df_disasters.columns:
                    st.write("#### Tendu00eancia Temporal de Desastres")

                    # Agrupar por ano
                    yearly_data = (
                        df_disasters.groupby("Year").size().reset_index(name="count")
                    )

                    # Gru00e1fico de linha por ano
                    fig = px.line(
                        yearly_data,
                        x="Year",
                        y="count",
                        title="Nu00famero de Desastres por Ano",
                        labels={"count": "Nu00famero de Desastres", "Year": "Ano"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Agrupar por ano e tipo de desastre
                    if "Disaster Type" in df_disasters.columns:
                        yearly_by_type = (
                            df_disasters.groupby(["Year", "Disaster Type"])
                            .size()
                            .reset_index(name="count")
                        )

                        fig = px.line(
                            yearly_by_type,
                            x="Year",
                            y="count",
                            color="Disaster Type",
                            title="Nu00famero de Desastres por Ano e Tipo",
                            labels={
                                "count": "Nu00famero de Desastres",
                                "Year": "Ano",
                                "Disaster Type": "Tipo de Desastre",
                            },
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Exibir dados brutos
            if st.checkbox("Mostrar dados brutos"):
                st.write(df_disasters)
        else:
            st.warning(
                "Nu00e3o foi possu00edvel carregar dados histu00f3ricos de desastres."
            )

    with tab2:
        st.header("Prediu00e7u00f5es do Modelo")

        if predictions is not None and not predictions.empty:
            st.subheader("u00daltimas Prediu00e7u00f5es")

            # Mostrar prediu00e7u00f5es em cards
            for idx, pred in predictions.iterrows():
                # Definir cor com base na probabilidade
                if pred["probability"] >= 0.7:
                    card_color = "#ffcccb"  # Light red
                elif pred["probability"] >= 0.4:
                    card_color = "#ffffcc"  # Light yellow
                else:
                    card_color = "#ccffcc"  # Light green

                st.markdown(
                    f"""
                <div style="padding: 15px; border-radius: 5px; background-color: {card_color}; margin-bottom: 10px;">
                    <h3>{pred['disaster_type']}</h3>
                    <p><strong>Probabilidade:</strong> {pred['probability']:.2f}</p>
                    <p><strong>Impacto Estimado:</strong> {pred['estimated_impact']}</p>
                    <p><strong>Mortalidade Prevista:</strong> {pred['predicted_mortality']}</p>
                    <p><strong>Pessoas Afetadas Previstas:</strong> {pred['predicted_affected']}</p>
                    <p><strong>Timestamp:</strong> {pred['timestamp']}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Gru00e1fico de probabilidades por tipo de desastre
            st.subheader("Probabilidades por Tipo de Desastre")
            fig = px.bar(
                predictions,
                x="disaster_type",
                y="probability",
                title="Probabilidade por Tipo de Desastre",
                color="probability",
                color_continuous_scale="RdYlGn_r",
                labels={
                    "probability": "Probabilidade",
                    "disaster_type": "Tipo de Desastre",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nu00e3o hu00e1 prediu00e7u00f5es disponu00edveis no momento.")

    with tab3:
        st.header("Modelo de Machine Learning")

        st.write(
            """
        ### Modelo de Redes Neurais para Prediu00e7u00e3o de Desastres
        
        O sistema utiliza uma Rede Neural Perceptron Multicamadas (MLP) para realizar tru00eas tipos de prediu00e7u00f5es:
        
        1. **Classificau00e7u00e3o Binu00e1ria de Alto Impacto**: Prever se um evento tem potencial para ser um desastre de alto impacto.
        2. **Prediu00e7u00e3o de Mortalidade**: Estimar o nu00famero potencial de mortes.
        3. **Prediu00e7u00e3o de Pessoas Afetadas**: Estimar o nu00famero potencial de pessoas afetadas.
        
        #### Arquitetura do Modelo
        - Camada de entrada: Recebe dados de sensores ambientais (temperatura, umidade, pressu00e3o, etc.)
        - Camadas ocultas: Processamento dos padru00f5es e relau00e7u00f5es nos dados
        - Camadas de sau00edda: Gerau00e7u00e3o das prediu00e7u00f5es finais
        
        #### Performance do Modelo
        """
        )

        # Criar dados mockados de performance de modelo
        performance = pd.DataFrame(
            {
                "Modelo": ["MLP - Binary", "MLP - Mortality", "MLP - Affected"],
                "Accuracy": [0.85, None, None],
                "Precision": [0.82, None, None],
                "Recall": [0.78, None, None],
                "F1-Score": [0.80, None, None],
                "MSE": [None, 0.15, 0.22],
                "MAE": [None, 0.12, 0.18],
                "R2": [None, 0.78, 0.75],
            }
        )

        st.dataframe(performance)

        st.write(
            """
        #### Recursos do Modelo
        - **Treinamento Contu00ednuo**: O modelo u00e9 continuamente refinado com novos dados.
        - **Validau00e7u00e3o Cruzada**: Garantia de generalizau00e7u00e3o e robustez.
        - **Integrau00e7u00e3o com Sensores**: Utiliza dados em tempo real para prediu00e7u00f5es.
        - **Sistema de Alertas**: Ativa alertas automaticamente com base nas prediu00e7u00f5es.
        """
        )

        # Simular feature importance
        st.subheader("Importu00e2ncia das Caracteru00edsticas")
        feature_importance = pd.DataFrame(
            {
                "Feature": [
                    "temperature",
                    "humidity",
                    "pressure",
                    "water_level",
                    "soil_moisture",
                    "vibration",
                    "rain_level",
                ],
                "Importance": [0.25, 0.18, 0.15, 0.22, 0.10, 0.05, 0.05],
            }
        )

        fig = px.bar(
            feature_importance,
            x="Feature",
            y="Importance",
            title="Importância das Características no Modelo",
            color="Importance",
            labels={"Importance": "Importância", "Feature": "Característica"},
        )
        st.plotly_chart(fig, use_container_width=True)


# Executar a função show()
if __name__ == "__main__":
    show()
