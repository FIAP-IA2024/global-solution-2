#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemplo de uso dos modelos de machine learning treinados para previsão de desastres naturais.

Este script demonstra como carregar e utilizar os modelos treinados para fazer previsões
sobre novos dados de desastres naturais.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar o diretório raiz ao path para permitir importações relativas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Definir diretórios para carregar modelos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')


def load_model(model_name, target_type):
    """
    Carrega um modelo salvo a partir do disco.
    
    Args:
        model_name (str): Nome do modelo (e.g., 'random_forest', 'gradient_boosting')
        target_type (str): Tipo de alvo (e.g., 'binary_high_impact', 'mortality')
        
    Returns:
        modelo carregado
    """
    # Criar nome do arquivo
    filename = f"{model_name}_{target_type}.pkl"
    filepath = os.path.join(MODELS_DIR, filename)
    
    # Verificar se o arquivo existe
    if not os.path.exists(filepath):
        print(f"Erro: Modelo {filepath} não encontrado.")
        print("Execute primeiro o script model_development.py para treinar os modelos.")
        return None
    
    # Carregar o modelo
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Modelo carregado com sucesso: {filepath}")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None


def prepare_sample_data():
    """
    Prepara dados de exemplo para demonstrar o uso dos modelos.
    
    Em um cenário real, estes seriam novos dados obtidos de sensores ou fontes externas.
    
    Returns:
        DataFrame com dados de exemplo
    """
    # Carregar um subconjunto dos dados processados como exemplo
    try:
        processed_data_file = os.path.join(BASE_DIR, 'results', 'processed_data', 'processed_disasters_data.csv')
        df = pd.read_csv(processed_data_file)
        
        # Selecionar apenas 5 amostras para demonstração
        sample_data = df.sample(5, random_state=42)
        
        # Remover as colunas alvo para simular dados novos
        prediction_features = sample_data.drop(columns=[
            'Total Deaths', 'Total Affected', 'High_Death_Count', 'Combined_Impact'
        ], errors='ignore')
        
        # Manter as colunas alvo apenas para comparação com as previsões
        actual_values = {
            'binary_high_impact': sample_data['High_Death_Count'] if 'High_Death_Count' in sample_data.columns else None,
            'mortality': sample_data['Total Deaths'] if 'Total Deaths' in sample_data.columns else None,
            'affected': sample_data['Total Affected'] if 'Total Affected' in sample_data.columns else None
        }
        
        return prediction_features, actual_values
    
    except FileNotFoundError:
        print(f"Erro: Arquivo de dados processados não encontrado.")
        print("Execute primeiro os scripts de pré-processamento e desenvolvimento de modelos.")
        return None, None
    except Exception as e:
        print(f"Erro ao preparar dados de exemplo: {e}")
        return None, None


def make_predictions(model, features, target_type):
    """
    Faz previsões usando o modelo carregado.
    
    Args:
        model: Modelo carregado
        features (DataFrame): Features para previsão
        target_type (str): Tipo de alvo
        
    Returns:
        Previsões
    """
    if model is None or features is None:
        return None
    
    try:
        # Fazer previsões
        predictions = model.predict(features)
        
        # Para modelos de classificação, obter também probabilidades se disponível
        probabilities = None
        if hasattr(model, 'predict_proba') and target_type == 'binary_high_impact':
            probabilities = model.predict_proba(features)[:, 1]
        
        return predictions, probabilities
    except Exception as e:
        print(f"Erro ao fazer previsões: {e}")
        return None, None


def display_results(features, predictions, probabilities, actual_values, target_type):
    """
    Exibe os resultados das previsões.
    
    Args:
        features (DataFrame): Features utilizadas
        predictions: Previsões feitas pelo modelo
        probabilities: Probabilidades (para classificação)
        actual_values: Valores reais (se disponíveis)
        target_type (str): Tipo de alvo
    """
    if predictions is None:
        return
    
    print(f"\n{'-'*80}")
    print(f"Resultados das previsões para: {target_type}")
    print(f"{'-'*80}")
    
    # Preparar DataFrame para exibição
    results_df = features[['Disaster Type', 'Country']].copy()
    
    # Adicionar previsões
    results_df['Predicted'] = predictions
    
    # Adicionar probabilidades para classificação
    if probabilities is not None:
        results_df['Probability'] = probabilities
    
    # Adicionar valores reais se disponíveis
    if actual_values[target_type] is not None:
        results_df['Actual'] = actual_values[target_type].values
    
    # Exibir resultados
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(results_df)
    print("\n")
    
    # Para classificação, exibir métricas adicionais
    if target_type == 'binary_high_impact' and 'Actual' in results_df.columns:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(results_df['Actual'], results_df['Predicted'])
        precision = precision_score(results_df['Actual'], results_df['Predicted'], zero_division=0)
        recall = recall_score(results_df['Actual'], results_df['Predicted'], zero_division=0)
        f1 = f1_score(results_df['Actual'], results_df['Predicted'], zero_division=0)
        
        print(f"Métricas (com base apenas nas amostras):")
        print(f"  Acurácia: {accuracy:.4f}")
        print(f"  Precisão: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Criar visualização
        plt.figure(figsize=(10, 6))
        if probabilities is not None:
            plt.bar(range(len(probabilities)), probabilities, color='skyblue')
            plt.axhline(y=0.5, color='red', linestyle='--')
            plt.xlabel('Amostra')
            plt.ylabel('Probabilidade de Alto Impacto')
            plt.title('Probabilidades de Alto Impacto Previstas')
            plt.xticks(range(len(probabilities)), results_df.index)
            plt.tight_layout()
            plt.show()
    
    # Para regressão, exibir métricas adicionais
    elif target_type in ['mortality', 'affected'] and 'Actual' in results_df.columns:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
        r2 = r2_score(results_df['Actual'], results_df['Predicted'])
        
        print(f"Métricas (com base apenas nas amostras):")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Criar visualização
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.7)
        plt.plot([min(results_df['Actual']), max(results_df['Actual'])], 
                 [min(results_df['Actual']), max(results_df['Actual'])], 'r--')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Previstos')
        plt.title(f'Valores Reais vs. Previstos para {target_type}')
        plt.tight_layout()
        plt.show()


def main():
    """
    Função principal para demonstrar o uso dos modelos treinados.
    """
    print("Demonstração de uso dos modelos de machine learning para previsão de desastres naturais\n")
    
    # 1. Preparar dados de exemplo
    features, actual_values = prepare_sample_data()
    
    if features is None:
        return
    
    # 2. Definir modelos e alvos para demonstração
    model_examples = [
        {'model_name': 'random_forest', 'target_type': 'binary_high_impact'},
        {'model_name': 'gradient_boosting_regressor', 'target_type': 'mortality'},
        {'model_name': 'random_forest_regressor', 'target_type': 'affected'}
    ]
    
    # 3. Para cada modelo, fazer previsões e exibir resultados
    for example in model_examples:
        model_name = example['model_name']
        target_type = example['target_type']
        
        # Carregar modelo
        model = load_model(model_name, target_type)
        
        if model is not None:
            # Fazer previsões
            predictions, probabilities = make_predictions(model, features, target_type)
            
            # Exibir resultados
            display_results(features, predictions, probabilities, actual_values, target_type)


if __name__ == "__main__":
    main()
