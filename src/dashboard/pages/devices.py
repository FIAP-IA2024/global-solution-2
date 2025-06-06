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
                # Criar card com componentes nativos do Streamlit
                with st.container():
                    # Aplicar CSS para estilizar o container como um card
                    st.markdown(
                        """
                        <style>
                        .device-card {
                            border: 1px solid #ddd;
                            border-radius: 10px;
                            padding: 20px;
                            background-color: white;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            margin-bottom: 20px;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Iniciar div do card
                    st.markdown('<div class="device-card">', unsafe_allow_html=True)
                    
                    # Nome do dispositivo como título
                    device_name = device.get('name', device.get('device_id', 'Desconhecido'))
                    st.subheader(device_name)
                    
                    # Status do dispositivo com cor
                    status = device.get('status', 'N/A')
                    st.markdown(f"**Status:** <span style='color: {status_color}; font-weight: bold;'>{status}</span>", unsafe_allow_html=True)
                    
                    # Informações do dispositivo
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown(f"**ID:** {device.get('device_id', 'N/A')}")
                        st.markdown(f"**Localização:** {device.get('location', 'N/A')}")
                    
                    with col_b:
                        st.markdown(f"**Bateria:** {battery}% {battery_icon}")
                        st.markdown(f"**Última atualização:**  \n{device.get('last_update', 'N/A')}")
                    
                    # Botões de ação
                    if st.button("👁️ Detalhes", key=f"detail_{device.get('device_id', idx)}"):
                        # Mostrar mais detalhes quando o botão for clicado
                        with st.expander(f"Detalhes do dispositivo {device_name}", expanded=True):
                            # Dados dos sensores se disponível
                            if "readings" in device:
                                st.subheader("Leituras de Sensores")
                                # Criar gráficos ou tabelas para as leituras
                                readings = device.get("readings", {})
                                if readings:
                                    # Converter leituras para DataFrame
                                    readings_df = pd.DataFrame([readings])
                                    st.dataframe(readings_df)
                                    
                                    # Gráfico de barras para valores dos sensores
                                    st.bar_chart(readings_df.T)
                            
                            # Histórico e estatísticas do dispositivo
                            st.subheader("Informações Detalhadas")
                            
                            # Colunas para informações adicionais
                            info_col1, info_col2 = st.columns(2)
                            with info_col1:
                                st.markdown(f"**Modelo:** {device.get('model', 'ESP32')}")
                                st.markdown(f"**Versão do Firmware:** {device.get('firmware', 'v1.0.2')}")
                                st.markdown(f"**Última Manutenção:** {device.get('last_maintenance', 'N/A')}")
                            
                            with info_col2:
                                st.markdown(f"**IP:** {device.get('ip', '192.168.1.X')}")
                                st.markdown(f"**MAC:** {device.get('mac', 'XX:XX:XX:XX:XX:XX')}")
                                st.markdown(f"**Instalado em:** {device.get('installation_date', 'N/A')}")
                                
                            # Opções de gerenciamento
                            st.subheader("Gerenciamento do Dispositivo")
                            col_act1, col_act2, col_act3 = st.columns(3)
                            with col_act1:
                                if st.button("Reiniciar Dispositivo", key=f"restart_{device.get('device_id', idx)}"):
                                    st.success("Comando de reinicialização enviado! (simulação)")
                            with col_act2:
                                if st.button("Atualizar Firmware", key=f"update_{device.get('device_id', idx)}"):
                                    st.info("Iniciando atualização de firmware... (simulação)")
                            with col_act3:
                                if st.button("Calibrar Sensores", key=f"calibrate_{device.get('device_id', idx)}"):
                                    st.success("Sensores calibrados com sucesso! (simulação)")
                    
                    # Fechar div do card
                    st.markdown('</div>', unsafe_allow_html=True)

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
