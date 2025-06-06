import pandas as pd
import numpy as np
import datetime
import os
import json

# Funções para carregar dados mock e, posteriormente, dados reais da API

def load_disaster_dataset():
    """
    Carrega o dataset de desastres ou retorna dados mockados se o arquivo não estiver disponível
    """
    try:
        # Tenta carregar o dataset real
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        dataset_path = os.path.join(base_dir, 'dataset.xlsx')
        
        if os.path.exists(dataset_path):
            df = pd.read_excel(dataset_path)
            print(f"Dataset carregado com sucesso: {df.shape[0]} registros")
            return df
        else:
            print(f"Arquivo não encontrado: {dataset_path}")
            # Se o arquivo não existir, retorna dados mockados
            return create_mock_disaster_data()
    except Exception as e:
        print(f"Erro ao carregar dataset: {e}")
        return create_mock_disaster_data()

def create_mock_disaster_data():
    """
    Cria dados mockados de desastres para desenvolvimento
    """
    np.random.seed(42)
    # Criar 100 registros de desastres
    data = {
        'Disaster Type': np.random.choice(['Flood', 'Earthquake', 'Tornado', 'Wildfire', 'Drought'], 100),
        'Country': np.random.choice(['Brazil', 'USA', 'Japan', 'Italy', 'India', 'China'], 100),
        'Year': np.random.randint(1980, 2025, 100),
        'Month': np.random.randint(1, 13, 100),
        'Total Deaths': np.random.exponential(100, 100).astype(int),
        'Total Affected': np.random.exponential(10000, 100).astype(int),
        'Total Damages (USD)': np.random.exponential(1000000, 100)
    }
    return pd.DataFrame(data)

def load_sensor_data():
    """
    Carrega dados de sensores a partir do arquivo gerado pelo simulador ESP32 ou gera dados mockados
    """
    try:
        # Tenta carregar dados do arquivo gerado pelo simulador
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        data_dir = os.path.join(base_dir, 'data')
        sensor_data_file = os.path.join(data_dir, 'sensor_data.json')
        
        if os.path.exists(sensor_data_file):
            with open(sensor_data_file, 'r') as f:
                data = json.load(f)
                
            # Converter readings para DataFrame
            if 'readings' in data and data['readings']:
                readings = pd.DataFrame()
                
                for reading in data['readings']:
                    if 'readings' in reading and 'timestamp' in reading and 'device_id' in reading:
                        # Extrair os dados de leitura para um dicionário
                        row = {'timestamp': reading['timestamp'], 'device_id': reading['device_id']}
                        row.update(reading['readings'])  # Adicionar todas as leituras de sensores
                        
                        # Adicionar ao DataFrame
                        readings = pd.concat([readings, pd.DataFrame([row])], ignore_index=True)
                
                # Converter timestamp para datetime
                readings['timestamp'] = pd.to_datetime(readings['timestamp'])
                
                return readings
        
        # Se não conseguiu carregar ou o arquivo está vazio, retornar dados mockados
        print("Arquivo de dados de sensores não encontrado ou vazio. Usando dados mockados.")
        return create_mock_sensor_data()
    except Exception as e:
        print(f"Erro ao carregar dados de sensores: {e}")
        return create_mock_sensor_data()

def create_mock_sensor_data():
    """
    Cria dados mockados de sensores para desenvolvimento
    """
    np.random.seed(42)
    dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=7), 
                        end=datetime.datetime.now(), 
                        freq='H')
    
    data = {
        'timestamp': dates,
        'temperature': np.random.normal(25, 3, len(dates)),
        'humidity': np.random.normal(60, 10, len(dates)),
        'pressure': np.random.normal(1013, 5, len(dates)),
        'water_level': np.random.normal(20, 8, len(dates)),
        'soil_moisture': np.random.normal(40, 15, len(dates)),
        'vibration': np.random.normal(50, 30, len(dates)),
        'rain_level': np.random.normal(5, 3, len(dates))}
    
    # Adicionar device_id aleatório para cada leitura
    device_ids = ['ESP32_01', 'ESP32_02', 'ESP32_03', 'ESP32_04']
    data['device_id'] = np.random.choice(device_ids, len(dates))
    
    return pd.DataFrame(data)

def load_alerts():
    """
    Carrega alertas a partir do arquivo gerado pelo simulador ESP32 ou gera dados mockados
    """
    try:
        # Tenta carregar dados do arquivo gerado pelo simulador
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        data_dir = os.path.join(base_dir, 'data')
        sensor_data_file = os.path.join(data_dir, 'sensor_data.json')
        
        if os.path.exists(sensor_data_file):
            with open(sensor_data_file, 'r') as f:
                data = json.load(f)
                
            # Converter alertas para DataFrame
            if 'alerts' in data and data['alerts']:
                # Adicionar um ID sequencial para cada alerta
                for i, alert in enumerate(data['alerts']):
                    alert['id'] = i + 1
                    
                alerts_df = pd.DataFrame(data['alerts'])
                
                # Converter timestamp para datetime
                if 'timestamp' in alerts_df.columns:
                    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
                
                return alerts_df
        
        # Se não conseguiu carregar ou o arquivo está vazio, retornar dados mockados
        return create_mock_alerts()
    except Exception as e:
        print(f"Erro ao carregar alertas: {e}")
        return create_mock_alerts()

def create_mock_alerts():
    """
    Cria dados mockados de alertas para desenvolvimento
    """
    alerts = [
        {"id": 1, "timestamp": datetime.datetime.now() - datetime.timedelta(hours=2), 
         "device_id": "ESP32_01", "type": "high_temperature", "message": "Temperatura acima do limiar", "severity": "high"},
        {"id": 2, "timestamp": datetime.datetime.now() - datetime.timedelta(hours=5), 
         "device_id": "ESP32_02", "type": "vibration", "message": "Vibração anormal detectada", "severity": "medium"},
        {"id": 3, "timestamp": datetime.datetime.now() - datetime.timedelta(hours=12), 
         "device_id": "ESP32_03", "type": "water_level", "message": "Nível de água elevado", "severity": "high"},
        {"id": 4, "timestamp": datetime.datetime.now() - datetime.timedelta(days=1), 
         "device_id": "ESP32_04", "type": "low_pressure", "message": "Baixa pressão atmosférica", "severity": "low"},
    ]
    return pd.DataFrame(alerts)

def load_devices():
    """
    Carrega informações de dispositivos a partir do arquivo gerado pelo simulador ESP32 ou gera dados mockados
    """
    try:
        # Tenta carregar dados do arquivo gerado pelo simulador
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        data_dir = os.path.join(base_dir, 'data')
        sensor_data_file = os.path.join(data_dir, 'sensor_data.json')
        
        if os.path.exists(sensor_data_file):
            with open(sensor_data_file, 'r') as f:
                data = json.load(f)
                
            # Converter dados de dispositivos para DataFrame
            if 'devices' in data and data['devices']:
                devices_list = []
                
                for device_id, device_info in data['devices'].items():
                    device = {'id': device_id}
                    device.update(device_info)
                    devices_list.append(device)
                
                devices_df = pd.DataFrame(devices_list)
                
                # Converter last_update para datetime
                if 'last_update' in devices_df.columns:
                    devices_df['last_update'] = pd.to_datetime(devices_df['last_update'])
                
                return devices_df
        
        # Se não conseguiu carregar ou o arquivo está vazio, retornar dados mockados
        return create_mock_devices()
    except Exception as e:
        print(f"Erro ao carregar dispositivos: {e}")
        return create_mock_devices()

def create_mock_devices():
    """
    Cria dados mockados de dispositivos para desenvolvimento
    """
    devices = [
        {"id": "ESP32_01", "name": "Sensor São Paulo", "location": "São Paulo, SP", 
         "status": "online", "last_update": datetime.datetime.now() - datetime.timedelta(minutes=5),
         "lat": -23.5505, "lon": -46.6333},
        {"id": "ESP32_02", "name": "Sensor Rio de Janeiro", "location": "Rio de Janeiro, RJ", 
         "status": "online", "last_update": datetime.datetime.now() - datetime.timedelta(minutes=10),
         "lat": -22.9068, "lon": -43.1729},
        {"id": "ESP32_03", "name": "Sensor Belo Horizonte", "location": "Belo Horizonte, MG", 
         "status": "offline", "last_update": datetime.datetime.now() - datetime.timedelta(hours=3),
         "lat": -19.9167, "lon": -43.9345},
        {"id": "ESP32_04", "name": "Sensor Salvador", "location": "Salvador, BA", 
         "status": "warning", "last_update": datetime.datetime.now() - datetime.timedelta(minutes=30),
         "lat": -12.9714, "lon": -38.5014},
    ]
    return pd.DataFrame(devices)

def load_prediction_results():
    """
    Carrega resultados de predição do modelo de ML a partir de dados reais
    ou gera dados mockados se os modelos não estiverem disponíveis
    """
    try:
        # Tenta carregar o dataset real para usar como base para predições
        df_disasters = load_disaster_dataset()
        
        # Verifica se temos dados reais para trabalhar
        if isinstance(df_disasters, pd.DataFrame) and not df_disasters.empty and 'Disaster Type' in df_disasters.columns:
            # Seleciona registros recentes ou amostra aleatória se não houver dados temporais recentes
            recent_data = df_disasters.sample(min(10, df_disasters.shape[0]))
            
            now = datetime.datetime.now()
            predictions = []
            
            # Gerar predições baseadas em dados reais
            for idx, row in recent_data.iterrows():
                disaster_type = row.get('Disaster Type', 'Unknown')
                region = row.get('Country', 'Unknown')
                
                # Simular uso de modelo ML para predições
                # Em produção, aqui seria carregado um modelo treinado
                mortality = int(row.get('Total Deaths', 0) * np.random.uniform(0.8, 1.2))
                affected = int(row.get('Total Affected', 0) * np.random.uniform(0.8, 1.2)) if 'Total Affected' in row else int(np.random.exponential(10000))
                
                # Calcular probabilidade com base nos dados históricos
                probability = np.random.uniform(0.3, 0.9)
                
                # Classificar impacto
                impact = 'high' if mortality > 100 or affected > 10000 else \
                         'medium' if mortality > 10 or affected > 1000 else 'low'
                
                predictions.append({
                    "timestamp": now - datetime.timedelta(hours=np.random.randint(1, 48)), 
                    "disaster_type": disaster_type, 
                    "region": region,
                    "probability": probability, 
                    "estimated_impact": impact,
                    "predicted_mortality": mortality,
                    "predicted_affected": affected
                })
            
            return pd.DataFrame(predictions)
    except Exception as e:
        print(f"Erro ao gerar previsões com base em dados reais: {e}")
        # Em caso de erro, retorna dados mockados
    
    # Fallback para dados mockados
    now = datetime.datetime.now()
    predictions = [
        {"timestamp": now - datetime.timedelta(hours=1), 
         "disaster_type": "Flood", 
         "region": "Southeast Asia",
         "probability": 0.75, 
         "estimated_impact": "high",
         "predicted_mortality": 150,
         "predicted_affected": 15000},
        {"timestamp": now - datetime.timedelta(hours=6), 
         "disaster_type": "Earthquake", 
         "region": "Japan",
         "probability": 0.25, 
         "estimated_impact": "medium",
         "predicted_mortality": 50,
         "predicted_affected": 5000},
        {"timestamp": now - datetime.timedelta(hours=12), 
         "disaster_type": "Landslide", 
         "region": "Brazil",
         "probability": 0.60, 
         "estimated_impact": "medium",
         "predicted_mortality": 30,
         "predicted_affected": 2000},
        {"timestamp": now - datetime.timedelta(days=1), 
         "disaster_type": "Wildfire", 
         "region": "Australia",
         "probability": 0.40, 
         "estimated_impact": "low",
         "predicted_mortality": 5,
         "predicted_affected": 500},
    ]
    return pd.DataFrame(predictions)
