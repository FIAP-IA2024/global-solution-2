#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Teste rápido de redes neurais para previsão de impacto de desastres naturais.
Versão otimizada para execução mais rápida com menos épocas e apenas MLP.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)

# Importar TensorFlow e Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configuração para execução mais rápida
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Reduzir verbosidade do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR

# Diretórios para modelos e resultados
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models', 'neural_networks_test')
REPORTS_DIR = os.path.join(BASE_DIR, 'results', 'reports')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots', 'models', 'neural_networks_test')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'results', 'processed_data')

# Função para garantir que diretórios existam
def ensure_dir(directory):
    """
    Garante que um diretório existe, criando-o se necessário.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# Criar diretórios necessários
for dir_path in [MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    ensure_dir(dir_path)

def load_data():
    """
    Carrega os dados processados para treinar as redes neurais.
    
    Returns:
        DataFrame com dados processados
    """
    try:
        data_file = os.path.join(PROCESSED_DATA_DIR, 'processed_disasters_data.csv')
        print(f"Carregando dados do arquivo: {data_file}")
        
        df = pd.read_csv(data_file)
        print(f"Dados carregados com sucesso: {df.shape[0]} linhas e {df.shape[1]} colunas")
        
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        sys.exit(1)

def prepare_data_for_neural_networks(df, target_column):
    """
    Prepara os dados para uso com redes neurais, realizando normalizações
    e transformações específicas.
    
    Args:
        df (DataFrame): Dados processados
        target_column (str): Nome da coluna alvo para previsão
    
    Returns:
        X_train, X_test, y_train, y_test, scaler_X, scaler_y (se regressão)
    """
    try:
        # Verificar se a coluna alvo existe
        if target_column not in df.columns:
            print(f"Erro: Coluna {target_column} não encontrada no DataFrame")
            return None, None, None, None, None, None, None
        
        # Determinar as features (todas as colunas exceto as de target)
        target_columns = ['High_Death_Count', 'Total Deaths', 'No. Affected', 'Total Affected']
        feature_columns = [col for col in df.columns if col not in target_columns]
        
        # Separar features e target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Dividir em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Dados já estão normalizados, então vamos pular a normalização adicional
        print(f"Garantindo que os dados estejam no formato correto para TensorFlow")
        
        # Converter para float32 para compatibilidade com TensorFlow
        X_train_scaled = X_train.astype('float32')
        X_test_scaled = X_test.astype('float32')
        
        # Para classificação binária, garantir que os dados estejam como float32
        if target_column == 'High_Death_Count':
            y_train = y_train.astype('float32')
            y_test = y_test.astype('float32')
        else:
            # Para regressão, garantir que os dados estejam como float32
            y_train = y_train.astype('float32')
            y_test = y_test.astype('float32')
        
        scaler_X = None
        scaler_y = None
        
        print(f"Dados preparados para {target_column}")
        print(f"  Treino: {X_train_scaled.shape[0]} amostras")
        print(f"  Teste: {X_test_scaled.shape[0]} amostras")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, scaler_y, feature_columns
    
    except Exception as e:
        print(f"Erro ao preparar dados para {target_column}: {e}")
        return None, None, None, None, None, None, None

def create_mlp_model(input_dim, output_dim, problem_type):
    """
    Cria um modelo MLP (Perceptron de Múltiplas Camadas) para classificação ou regressão.
    Versão simplificada para execução mais rápida.
    
    Args:
        input_dim (int): Dimensão de entrada (número de features)
        output_dim (int): Dimensão de saída (1 para regressão, 2 para classificação binária)
        problem_type (str): Tipo de problema ('classification' ou 'regression')
    
    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
    ])
    
    # Camada de saída e compilação específica para o tipo de problema
    if problem_type == 'classification':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:  # regression
        model.add(Dense(1, activation='linear'))
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    
    return model

def train_neural_network(model, X_train, X_test, y_train, y_test, model_name, target_type):
    """
    Treina uma rede neural e salva o melhor modelo.
    Versão simplificada para execução mais rápida.
    
    Args:
        model: Modelo Keras compilado
        X_train, X_test, y_train, y_test: Dados de treino e teste
        model_name (str): Nome do modelo para salvar
        target_type (str): Tipo de alvo (binary_high_impact, mortality, affected)
    
    Returns:
        História de treinamento, melhor modelo
    """
    # Configurar callbacks para treinamento
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{target_type}.h5")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    print(f"Iniciando treinamento do modelo {model_name} para {target_type}...")
    start_time = time.time()
    
    # Treinar o modelo com menos épocas para teste rápido
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Reduzido para execução mais rápida
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Treinamento concluído em {training_time:.2f} segundos")
    
    # Carregar o melhor modelo
    best_model = load_model(model_path)
    print(f"Melhor modelo carregado de {model_path}")
    
    return history, best_model

def evaluate_model(model, X_test, y_test, scaler_y, problem_type, model_name, target_type):
    """
    Avalia um modelo neural e retorna as métricas.
    
    Args:
        model: Modelo treinado
        X_test, y_test: Dados de teste
        scaler_y: Scaler para desfazer normalização (None para classificação)
        problem_type: 'classification' ou 'regression'
        model_name, target_type: Informações do modelo
    
    Returns:
        Dictionary com métricas de avaliação
    """
    print(f"\nAvaliando modelo {model_name} para {target_type}...")
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    metrics = {}
    
    if problem_type == 'classification':
        # Transformar probabilidades em classes
        y_pred_class = (y_pred > 0.5).astype(int).flatten()
        
        # Calcular métricas
        metrics['accuracy'] = accuracy_score(y_test, y_pred_class)
        metrics['precision'] = precision_score(y_test, y_pred_class)
        metrics['recall'] = recall_score(y_test, y_pred_class)
        metrics['f1'] = f1_score(y_test, y_pred_class)
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred.flatten())
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_class)
        
        # Exibir resultados
        print(f"Métricas de classificação para {model_name} - {target_type}:")
        print(f"  Acurácia: {metrics['accuracy']:.4f}")
        print(f"  Precisão: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    else:  # regression
        # Dados já estão normalizados, não precisamos desnormalizar
        y_test_original = y_test
        y_pred = y_pred.flatten()
        
        # Calcular métricas
        metrics['mse'] = mean_squared_error(y_test_original, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_test_original, y_pred)
        metrics['r2'] = r2_score(y_test_original, y_pred)
        
        # Exibir resultados
        print(f"Métricas de regressão para {model_name} - {target_type}:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    return metrics

def main():
    """
    Função principal para executar o teste rápido da rede neural MLP.
    """
    print("\n================================================================")
    print("TESTE RÁPIDO DE REDE NEURAL - Versão otimizada para execução rápida")
    print("================================================================\n")
    
    # Carregar os dados processados
    df = load_data()
    
    # Definir alvos para teste rápido usando os nomes corretos das colunas
    # Mapeamento entre nomes internos e nomes das colunas no CSV
    target_mapping = {
        'binary_high_impact': 'High_Death_Count',  # Coluna de classificação binária
        'mortality': 'Total Deaths',             # Regressão para mortalidade
        'affected': 'No. Affected'               # Regressão para pessoas afetadas
    }
    
    targets = list(target_mapping.values())  # Usar os nomes das colunas reais do CSV
    
    for target_type in targets:
        print(f"\n================================================================")
        print(f"Desenvolvendo modelo para: {target_type}")
        print(f"================================================================\n")
        
        # Preparar os dados para o alvo específico
        problem_type = 'classification' if target_type == 'High_Death_Count' else 'regression'
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names = prepare_data_for_neural_networks(df, target_type)
        
        if X_train is None or X_test is None:
            print(f"Erro ao preparar dados para {target_type}. Pulando...")
            continue
        
        # Definir dimensões de entrada e saída
        input_dim = X_train.shape[1]
        output_dim = 1  # Sempre 1 nesta versão simplificada
        
        # Criar o modelo MLP
        model = create_mlp_model(input_dim, output_dim, problem_type)
        
        # Treinar o modelo
        history, trained_model = train_neural_network(
            model, X_train, X_test, y_train, y_test, "mlp", target_type
        )
        
        # Avaliar o modelo
        metrics = evaluate_model(
            trained_model, X_test, y_test, scaler_y, problem_type, "mlp", target_type
        )
    
    print("\n================================================================")
    print("Teste rápido de rede neural concluído com sucesso!")
    print("Os modelos foram salvos no diretório de modelos.")
    print("================================================================\n")

if __name__ == '__main__':
    # Importar aqui para não afetar o tempo de inicialização
    import time
    main()
