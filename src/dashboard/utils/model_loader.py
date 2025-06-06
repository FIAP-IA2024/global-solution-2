"""
Utilitário para carregar modelos de ML treinados e fazer predições
Global Solution 2025.1 - FIAP
"""

import os
import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime

# Diretórios onde os modelos treinados são armazenados
def get_model_dir():
    """Retorna o diretório onde os modelos estão armazenados"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(base_dir, 'results', 'models')

def load_model(model_file):
    """
    Carrega um modelo de ML salvo
    
    Args:
        model_file: Nome do arquivo do modelo
        
    Returns:
        Modelo carregado ou None se falhar
    """
    try:
        model_path = os.path.join(get_model_dir(), model_file)
        
        if not os.path.exists(model_path):
            print(f"Arquivo de modelo não encontrado: {model_path}")
            return None
        
        # Tenta carregar o modelo usando diferentes formatos
        try:
            # Tenta primeiro com pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except:
            try:
                # Tenta com joblib
                model = joblib.load(model_path)
            except:
                # Tenta com formato específico de TensorFlow/Keras
                try:
                    from tensorflow import keras
                    model = keras.models.load_model(model_path)
                except:
                    print(f"Não foi possível carregar o modelo {model_path} em nenhum formato conhecido")
                    return None
        
        print(f"Modelo {model_file} carregado com sucesso")
        return model
    
    except Exception as e:
        print(f"Erro ao carregar modelo {model_file}: {e}")
        return None

def load_all_available_models():
    """
    Carrega todos os modelos disponíveis
    
    Returns:
        Dicionário com os modelos carregados
    """
    models = {}
    model_dir = get_model_dir()
    
    # Verifica se o diretório existe
    if not os.path.exists(model_dir):
        print(f"Diretório de modelos não encontrado: {model_dir}")
        return models
    
    # Lista todos os arquivos no diretório de modelos
    try:
        model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) 
                      and (f.endswith('.pkl') or f.endswith('.joblib') or f.endswith('.h5') or f.endswith('.model'))]
        
        # Tenta carregar cada arquivo de modelo
        for model_file in model_files:
            model_name = os.path.splitext(model_file)[0]  # Remove a extensão
            model = load_model(model_file)
            
            if model is not None:
                models[model_name] = model
    except Exception as e:
        print(f"Erro ao listar modelos: {e}")
    
    return models

def predict_with_model(model, features, model_type='classification'):
    """
    Faz predições com um modelo carregado
    
    Args:
        model: Modelo ML carregado
        features: Características para predição
        model_type: Tipo de modelo ('classification' ou 'regression')
        
    Returns:
        Resultado da predição
    """
    try:
        # Verifica o tipo de entrada esperada pelo modelo e ajusta se necessário
        if hasattr(model, 'predict_proba'):
            # Para classificadores que suportam probabilidades
            pred = model.predict_proba(features)
            if pred.shape[1] >= 2:  # Se tiver mais de uma classe
                return {'class_probabilities': pred, 'class': model.predict(features)}
            else:
                return {'probability': pred.flatten(), 'class': model.predict(features)}
        else:
            # Para regressores ou classificadores sem predict_proba
            pred = model.predict(features)
            
            if model_type == 'classification':
                return {'class': pred}
            else:  # regression
                return {'value': pred}
    except Exception as e:
        print(f"Erro ao fazer predição: {e}")
        return None

def get_feature_importance(model):
    """
    Tenta extrair importância de características do modelo
    
    Args:
        model: Modelo ML carregado
        
    Returns:
        DataFrame com características e suas importâncias ou None se não for possível
    """
    try:
        # Verificar se o modelo tem feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
            else:
                # Se não tiver os nomes, usar índices
                features = [f"feature_{i}" for i in range(len(importances))]
                
            return pd.DataFrame({'feature': features, 'importance': importances})
            
        # Verificar se é um modelo linear com coef_
        elif hasattr(model, 'coef_'):
            coefs = model.coef_
            
            # Verificar se é um coeficiente por classe (multiclasse)
            if len(coefs.shape) > 1:
                # Pegar a média absoluta dos coeficientes entre classes
                importances = np.mean(np.abs(coefs), axis=0)
            else:
                importances = np.abs(coefs)
                
            if hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
            else:
                features = [f"feature_{i}" for i in range(len(importances))]
                
            return pd.DataFrame({'feature': features, 'importance': importances})
            
        else:
            return None
    
    except Exception as e:
        print(f"Erro ao extrair importância de características: {e}")
        return None

def simulate_ml_prediction(sensor_data, num_predictions=5):
    """
    Simula predições de ML baseadas em dados de sensores reais
    Esta função simula o comportamento de um modelo ML real
    
    Args:
        sensor_data: DataFrame com dados de sensores
        num_predictions: Número de predições a gerar
        
    Returns:
        DataFrame com predições
    """
    # Usar dados de sensores para criar predições mais realistas
    predictions = []
    
    # Tipos de desastres para modelar
    disaster_types = ['Flood', 'Earthquake', 'Landslide', 'Wildfire', 'Storm', 'Drought']
    
    # Verificar se temos dados de sensores suficientes
    if sensor_data is None or len(sensor_data) < 10:
        # Gerar predições completamente sintéticas
        for _ in range(num_predictions):
            disaster_type = np.random.choice(disaster_types)
            
            # Calcular probabilidade baseada no tipo de desastre
            probability = np.random.uniform(0.3, 0.9)
            
            # Gerar valores de impacto
            casualties = int(np.random.exponential(50))
            affected = int(np.random.exponential(5000))
            
            # Determinar impacto com base em casualties e affected
            impact = 'high' if casualties > 100 or affected > 10000 else \
                     'medium' if casualties > 10 or affected > 1000 else 'low'
                     
            # Região aleatória
            regions = ['São Paulo', 'Rio de Janeiro', 'Minas Gerais', 'Amazonas', 'Pará', 
                      'Santa Catarina', 'Rio Grande do Sul', 'Bahia', 'Ceará', 'Paraná']
            region = np.random.choice(regions)
            
            predictions.append({
                'timestamp': datetime.now(),
                'disaster_type': disaster_type,
                'region': region,
                'probability': probability,
                'estimated_impact': impact,
                'predicted_mortality': casualties,
                'predicted_affected': affected
            })
    else:
        # Usar os dados de sensores para criar predições mais realistas
        # Seleciona algumas amostras dos dados de sensores
        samples = sensor_data.sample(min(num_predictions, len(sensor_data)))
        
        for _, sample in samples.iterrows():
            # Selecionar tipo de desastre com base nos valores dos sensores
            if 'temperature' in sample and 'humidity' in sample and 'rain_level' in sample:
                # Lógica simples para determinar tipo de desastre mais provável
                if sample['temperature'] > 30 and sample['humidity'] < 30:
                    disaster_type = 'Wildfire'
                    probability = min(0.9, sample['temperature'] / 50 + (100 - sample['humidity']) / 100)
                elif sample['rain_level'] > 15 or sample['water_level'] > 30:
                    disaster_type = 'Flood'
                    probability = min(0.9, sample['rain_level'] / 30 + sample['water_level'] / 50)
                elif 'vibration' in sample and sample['vibration'] > 100:
                    disaster_type = 'Earthquake'
                    probability = min(0.9, sample['vibration'] / 200)
                elif 'soil_moisture' in sample and sample['soil_moisture'] < 20 and sample['rain_level'] < 5:
                    disaster_type = 'Drought'
                    probability = min(0.9, (20 - sample['soil_moisture']) / 20)
                elif 'pressure' in sample and sample['pressure'] < 1000:
                    disaster_type = 'Storm'
                    probability = min(0.9, (1013 - sample['pressure']) / 30)
                else:
                    disaster_type = np.random.choice(disaster_types)
                    probability = np.random.uniform(0.3, 0.7)
            else:
                disaster_type = np.random.choice(disaster_types)
                probability = np.random.uniform(0.3, 0.7)
            
            # Calcular valores de impacto com base no tipo e probabilidade
            casualties_factor = probability * (2 if disaster_type in ['Earthquake', 'Flood'] else 1)
            casualties = int(np.random.exponential(50 * casualties_factor))
            
            affected_factor = probability * (3 if disaster_type in ['Flood', 'Drought', 'Storm'] else 1)
            affected = int(np.random.exponential(5000 * affected_factor))
            
            # Determinar impacto
            impact = 'high' if casualties > 100 or affected > 10000 or probability > 0.8 else \
                     'medium' if casualties > 10 or affected > 1000 or probability > 0.5 else 'low'
            
            # Região baseada no device_id, se disponível
            if 'device_id' in sample:
                if sample['device_id'] == 'ESP32_01':
                    region = 'São Paulo'
                elif sample['device_id'] == 'ESP32_02':
                    region = 'Rio de Janeiro' 
                elif sample['device_id'] == 'ESP32_03':
                    region = 'Belo Horizonte'
                elif sample['device_id'] == 'ESP32_04':
                    region = 'Salvador'
                else:
                    region = 'Brasil'
            else:
                regions = ['São Paulo', 'Rio de Janeiro', 'Minas Gerais', 'Amazonas']
                region = np.random.choice(regions)
            
            predictions.append({
                'timestamp': datetime.now(),
                'disaster_type': disaster_type,
                'region': region,
                'probability': probability,
                'estimated_impact': impact,
                'predicted_mortality': casualties,
                'predicted_affected': affected
            })
    
    return pd.DataFrame(predictions)
