import pandas as pd
import numpy as np
import datetime
import os

# Funções para carregar dados mock e, posteriormente, dados reais da API

def load_disaster_dataset():
    """
    Carrega o dataset de desastres ou retorna dados mockados se o arquivo não estiver disponível
    """
    try:
        # Tenta carregar o dataset real
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        dataset_path = os.path.join(base_dir, 'data', 'disasters_dataset.xlsx')
        
        if os.path.exists(dataset_path):
            return pd.read_excel(dataset_path)
        else:
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
    Carrega dados de sensores a partir da API ou gera dados mockados
    """
    # Na aplicação real, isso buscaria dados da API
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
        'rain_level': np.random.normal(5, 3, len(dates))
    }
    
    return pd.DataFrame(data)

def load_alerts():
    """
    Carrega alertas a partir da API ou gera dados mockados
    """
    # Na aplicação real, isso buscaria alertas da API
    alerts = [
        {"id": 1, "timestamp": datetime.datetime.now() - datetime.timedelta(hours=2), 
         "type": "high_temperature", "message": "Temperatura acima do limiar", "severity": "high"},
        {"id": 2, "timestamp": datetime.datetime.now() - datetime.timedelta(hours=5), 
         "type": "vibration", "message": "Vibração anormal detectada", "severity": "medium"},
        {"id": 3, "timestamp": datetime.datetime.now() - datetime.timedelta(hours=12), 
         "type": "water_level", "message": "Nível de água elevado", "severity": "high"},
        {"id": 4, "timestamp": datetime.datetime.now() - datetime.timedelta(days=1), 
         "type": "low_pressure", "message": "Baixa pressão atmosférica", "severity": "low"},
    ]
    return pd.DataFrame(alerts)

def load_devices():
    """
    Carrega informações de dispositivos a partir da API ou gera dados mockados
    """
    # Na aplicação real, isso buscaria dispositivos da API
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
    Carrega resultados de predição do modelo de ML a partir da API ou gera dados mockados
    """
    # Na aplicação real, isso buscaria resultados de predição da API
    now = datetime.datetime.now()
    predictions = [
        {"timestamp": now - datetime.timedelta(hours=1), 
         "disaster_type": "Flood", 
         "probability": 0.75, 
         "estimated_impact": "high",
         "predicted_mortality": 150,
         "predicted_affected": 15000},
        {"timestamp": now - datetime.timedelta(hours=6), 
         "disaster_type": "Earthquake", 
         "probability": 0.25, 
         "estimated_impact": "medium",
         "predicted_mortality": 50,
         "predicted_affected": 5000},
        {"timestamp": now - datetime.timedelta(hours=12), 
         "disaster_type": "Landslide", 
         "probability": 0.60, 
         "estimated_impact": "medium",
         "predicted_mortality": 30,
         "predicted_affected": 2000},
        {"timestamp": now - datetime.timedelta(days=1), 
         "disaster_type": "Wildfire", 
         "probability": 0.40, 
         "estimated_impact": "low",
         "predicted_mortality": 5,
         "predicted_affected": 500},
    ]
    return pd.DataFrame(predictions)
