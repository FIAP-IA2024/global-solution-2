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
        "Análise de dados históricos de desastres e predições do modelo de Machine Learning."
    )
    
    # Adicionar spinner global no topo
    with st.spinner("Carregando dados e preparando análises..."):
        # Carregar dados de desastres histu00f3ricos
        df_disasters = load_disaster_dataset()
        predictions = load_prediction_results()

    # Tabs para separar os diferentes tipos de análise
    tab1, tab2, tab3 = st.tabs(
        ["Dados Históricos", "Predições", "Modelo de ML"]
    )

    with tab1:
        st.header("Análise de Dados Históricos")

        if df_disasters is not None and not df_disasters.empty:
            # Usar progress bar para mostrar o carregamento dos resumos
            with st.spinner("Carregando resumos estatísticos..."):
                # Resumo estatístico dos dados
                st.subheader("Resumo Estatístico")

            # Verificar as colunas disponíveis no dataframe
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

                # Mostrar gráfico de barras por tipo de desastre
                st.write("#### Impacto por Tipo de Desastre")

                impact_metric = st.selectbox(
                    "Selecione o métrico de impacto",
                    options=["Total Deaths", "Total Affected"],
                )
                
                # Spinner para o gráfico de barras
                with st.spinner("Gerando gráfico de impacto..."):
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

                # Análise temporal
                if "Year" in df_disasters.columns:
                    st.write("#### Tendência Temporal de Desastres")
                    
                    # Spinner para a análise temporal
                    with st.spinner("Calculando tendências temporais..."):
                        # Agrupar por ano
                        yearly_data = (
                            df_disasters.groupby("Year").size().reset_index(name="count")
                        )

                        # Gráfico de linha por ano
                        fig = px.line(
                            yearly_data,
                            x="Year",
                            y="count",
                            title="Número de Desastres por Ano",
                            labels={"count": "Número de Desastres", "Year": "Ano"},
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
                                title="Número de Desastres por Ano e Tipo",
                                labels={
                                    "count": "Número de Desastres",
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
                "Não foi possível carregar dados históricos de desastres."
            )

    with tab2:
        st.header("Predições do Modelo")

        if predictions is not None and not predictions.empty:
            # Spinner para predições
            with st.spinner("Carregando predições do modelo..."):
                st.subheader("Últimas Predições")

            # Mostrar predições em cards
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

            # Gráfico de probabilidades por tipo de desastre
            st.subheader("Probabilidades por Tipo de Desastre")
            
            # Spinner para gráfico de probabilidades
            with st.spinner("Gerando gráfico de probabilidades..."):
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
            st.warning("Não há predições disponíveis no momento.")

    with tab3:
        st.header("Modelo de Machine Learning")
        
        with st.spinner("Carregando informações do modelo..."):
            st.write(
                """
            ### Modelo de Redes Neurais para Predição de Desastres
        
        O sistema utiliza uma Rede Neural Perceptron Multicamadas (MLP) para realizar três tipos de predições:
        
        1. **Classificação do Tipo de Desastre**: Baseado nos dados dos sensores, o modelo prevê o tipo de desastre que pode ocorrer (inundação, deslizamento, incêndio ou terremoto)
        
        2. **Severidade do Desastre**: O modelo estima a gravidade do possível desastre em uma escala de 1-5
        
        3. **Probabilidade de Ocorrência**: Estimativa da chance do desastre realmente acontecer
        
        #### Arquitetura do Modelo
        - Camada de entrada: Recebe dados de sensores ambientais (temperatura, umidade, pressão, etc.)
        - Camadas ocultas: Processamento dos padrões e relações nos dados
        - Camadas de saída: Geração das predições finais
        
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
        - **Treinamento Contínuo**: O modelo é continuamente refinado com novos dados.
        - **Validação Cruzada**: Garantia de generalização e robustez.
        - **Integração com Sensores**: Utiliza dados em tempo real para predições.
        - **Sistema de Alertas**: Ativa alertas automaticamente com base nas predições.
        """
        )

        # Simular feature importance
        st.subheader("Importância das Características")
        
        # Spinner para o gráfico de feature importance
        with st.spinner("Calculando importância de características..."):
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
