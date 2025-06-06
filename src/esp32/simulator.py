#!/usr/bin/env python
"""
ESP32 Sensor Simulator
Global Solution 2025.1 - FIAP

Este script simula dados de sensores ESP32 para monitoramento de desastres naturais
e armazena os dados em um arquivo JSON que o dashboard pode consumir.
"""

import os
import json
import time
import random
import datetime
import numpy as np
from pathlib import Path

# Configuração de diretórios
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
SENSOR_DATA_FILE = DATA_DIR / "sensor_data.json"

# Configuração dos dispositivos simulados
DEVICE_IDS = ["ESP32_01", "ESP32_02", "ESP32_03", "ESP32_04"]
DEVICE_LOCATIONS = {
    "ESP32_01": {"name": "Sensor São Paulo", "location": "São Paulo, SP", "lat": -23.5505, "lon": -46.6333},
    "ESP32_02": {"name": "Sensor Rio de Janeiro", "location": "Rio de Janeiro, RJ", "lat": -22.9068, "lon": -43.1729},
    "ESP32_03": {"name": "Sensor Belo Horizonte", "location": "Belo Horizonte, MG", "lat": -19.9167, "lon": -43.9345},
    "ESP32_04": {"name": "Sensor Salvador", "location": "Salvador, BA", "lat": -12.9714, "lon": -38.5014}
}

# Parâmetros para simulação de sensores
SENSOR_PARAMS = {
    "temperature": {"mean": 25, "std": 3, "unit": "°C"},
    "humidity": {"mean": 60, "std": 10, "unit": "%"},
    "pressure": {"mean": 1013, "std": 5, "unit": "hPa"},
    "water_level": {"mean": 20, "std": 8, "unit": "cm"},
    "soil_moisture": {"mean": 40, "std": 15, "unit": "%"},
    "vibration": {"mean": 50, "std": 30, "unit": "Hz"},
    "rain_level": {"mean": 5, "std": 3, "unit": "mm"}
}

# Limiares para alertas (correspondem aos valores no código ESP32 real)
ALERT_THRESHOLDS = {
    "temperature_high": 35.0,
    "temperature_low": 5.0,
    "humidity_high": 90.0,
    "humidity_low": 20.0,
    "pressure_low": 1000.0,
    "water_level_high": 10.0,
    "vibration_high": 500
}

# Função para simular um valor de sensor com base nos parâmetros
def simulate_sensor_value(sensor_name):
    """Simula um valor de sensor com base nos parâmetros configurados"""
    params = SENSOR_PARAMS[sensor_name]
    return np.random.normal(params["mean"], params["std"])

# Função para determinar se uma condição de alerta foi acionada
def check_alert_condition(readings):
    """Verifica se alguma condição de alerta foi acionada com base nas leituras"""
    alerts = []
    
    if readings["temperature"] > ALERT_THRESHOLDS["temperature_high"]:
        alerts.append({
            "type": "high_temperature",
            "message": f"Temperatura alta detectada: {readings['temperature']:.1f}°C",
            "severity": "high"
        })
    elif readings["temperature"] < ALERT_THRESHOLDS["temperature_low"]:
        alerts.append({
            "type": "low_temperature",
            "message": f"Temperatura baixa detectada: {readings['temperature']:.1f}°C",
            "severity": "medium"
        })
        
    if readings["humidity"] > ALERT_THRESHOLDS["humidity_high"]:
        alerts.append({
            "type": "high_humidity",
            "message": f"Umidade alta detectada: {readings['humidity']:.1f}%",
            "severity": "medium"
        })
    elif readings["humidity"] < ALERT_THRESHOLDS["humidity_low"]:
        alerts.append({
            "type": "low_humidity",
            "message": f"Umidade baixa detectada: {readings['humidity']:.1f}%",
            "severity": "low"
        })
        
    if readings["pressure"] < ALERT_THRESHOLDS["pressure_low"]:
        alerts.append({
            "type": "low_pressure",
            "message": f"Pressão atmosférica baixa: {readings['pressure']:.1f}hPa - possível tempestade",
            "severity": "high"
        })
        
    if readings["water_level"] > ALERT_THRESHOLDS["water_level_high"]:
        alerts.append({
            "type": "high_water_level",
            "message": f"Nível de água elevado: {readings['water_level']:.1f}cm - risco de inundação",
            "severity": "critical"
        })
        
    if readings["vibration"] > ALERT_THRESHOLDS["vibration_high"]:
        alerts.append({
            "type": "high_vibration",
            "message": f"Vibração anormal detectada: {readings['vibration']:.1f}Hz - possível terremoto",
            "severity": "critical"
        })
        
    # Simulação de chuva forte
    if readings["rain_level"] > 20:
        alerts.append({
            "type": "heavy_rain",
            "message": f"Chuva forte detectada: {readings['rain_level']:.1f}mm - risco de inundação",
            "severity": "high"
        })
        
    return alerts

# Função para simular leituras de sensores para um dispositivo
def simulate_device_reading(device_id):
    """Simula uma leitura completa de sensores para um dispositivo"""
    readings = {sensor: simulate_sensor_value(sensor) for sensor in SENSOR_PARAMS.keys()}
    
    # Adicionar status do dispositivo (online/offline/warning)
    status = "online"
    if random.random() < 0.05:  # 5% de chance de estar offline
        status = "offline"
    elif random.random() < 0.10:  # 10% de chance de estar em status de warning
        status = "warning"
    
    # Gerar alertas com base nas leituras
    alerts = check_alert_condition(readings)
    
    # Montar o objeto de leitura completo
    reading = {
        "device_id": device_id,
        "device_info": DEVICE_LOCATIONS[device_id],
        "timestamp": datetime.datetime.now().isoformat(),
        "readings": readings,
        "status": status,
        "alerts": alerts
    }
    
    return reading

def load_existing_data():
    """Carrega os dados existentes do arquivo JSON"""
    if SENSOR_DATA_FILE.exists():
        try:
            with open(SENSOR_DATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Erro ao decodificar arquivo {SENSOR_DATA_FILE}, criando novo.")
    return {"devices": {}, "readings": [], "alerts": []}

def save_data(data):
    """Salva os dados no arquivo JSON"""
    with open(SENSOR_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Dados salvos em {SENSOR_DATA_FILE}")

def simulate_single_round():
    """Simula uma rodada de leituras para todos os dispositivos"""
    # Carregar dados existentes
    data = load_existing_data()
    
    # Limitar o número de leituras armazenadas (manter apenas as últimas 1000)
    if len(data["readings"]) > 1000:
        data["readings"] = data["readings"][-1000:]
    
    # Simular leituras para cada dispositivo
    for device_id in DEVICE_IDS:
        reading = simulate_device_reading(device_id)
        
        # Atualizar informações do dispositivo
        data["devices"][device_id] = {
            "name": DEVICE_LOCATIONS[device_id]["name"],
            "location": DEVICE_LOCATIONS[device_id]["location"],
            "lat": DEVICE_LOCATIONS[device_id]["lat"],
            "lon": DEVICE_LOCATIONS[device_id]["lon"],
            "status": reading["status"],
            "last_update": reading["timestamp"]
        }
        
        # Adicionar leitura ao histórico
        data["readings"].append({
            "device_id": device_id,
            "timestamp": reading["timestamp"],
            "readings": reading["readings"]
        })
        
        # Adicionar alertas, se houver
        for alert in reading["alerts"]:
            alert_entry = alert.copy()
            alert_entry["device_id"] = device_id
            alert_entry["timestamp"] = reading["timestamp"]
            data["alerts"].append(alert_entry)
    
    # Limitar o número de alertas armazenados (manter apenas os últimos 100)
    if len(data["alerts"]) > 100:
        data["alerts"] = data["alerts"][-100:]
    
    # Salvar os dados
    save_data(data)
    
    return data

def print_simulation_summary(data):
    """Imprime um resumo dos dados simulados"""
    num_devices = len(data["devices"])
    num_readings = len(data["readings"])
    num_alerts = len(data["alerts"])
    
    print(f"\n=== Simulação ESP32 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Dispositivos ativos: {num_devices}")
    print(f"Total de leituras: {num_readings}")
    print(f"Total de alertas: {num_alerts}")
    
    # Exibir os alertas mais recentes
    if num_alerts > 0:
        print("\n--- Alertas Recentes ---")
        for alert in data["alerts"][-5:]:
            print(f"{alert['timestamp']}: {alert['message']} ({alert['severity']})")

def simulate_continuous(interval=60):
    """Executa a simulação continuamente com um intervalo especificado"""
    print(f"Iniciando simulação ESP32. Os dados serão salvos em {SENSOR_DATA_FILE}")
    print(f"Intervalo de simulação: {interval} segundos")
    print("Pressione Ctrl+C para encerrar a simulação")
    
    try:
        while True:
            data = simulate_single_round()
            print_simulation_summary(data)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nSimulação ESP32 encerrada pelo usuário")

if __name__ == "__main__":
    # Criar diretório de dados se não existir
    DATA_DIR.mkdir(exist_ok=True)
    
    # Iniciar simulação contínua com intervalo de 60 segundos
    simulate_continuous(60)
