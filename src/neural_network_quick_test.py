#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Teste ru00e1pido de redes neurais para previsu00e3o de impacto de desastres naturais.
Versu00e3o otimizada para execuu00e7u00e3o mais ru00e1pida com menos u00e9pocas e apenas MLP.
"""

import os
import sys
import time
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

# Configurau00e7u00e3o para execuu00e7u00e3o mais ru00e1pida
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Reduzir verbosidade do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR

# Diretu00f3rios para modelos e resultados
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models', 'neural_networks_test')
REPORTS_DIR = os.path.join(BASE_DIR, 'results', 'reports')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots', 'models', 'neural_networks_test')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'results', 'processed_data')

# Funu00e7u00e3o para garantir que diretu00f3rios existam
def ensure_dir(directory):
    """
    Garante que um diretu00f3rio existe, criando-o se necessu00e1rio.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# Criar diretu00f3rios necessu00e1rios
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
    Prepara os dados para uso com redes neurais, realizando pru00e9-processamento
    e conversu00f5es necessu00e1rias.
    
    Args:
        df (DataFrame): Dados processados
        target_column (str): Nome da coluna alvo para previsu00e3o
    
    Returns:
        X_train, X_test, y_train, y_test, scaler_X, scaler_y (se regressu00e3o)
    """
    try:
        # Verificar se a coluna alvo existe
        if target_column not in df.columns:
            print(f"Erro: Coluna {target_column} nu00e3o encontrada no DataFrame")
            return None, None, None, None, None, None, None
        
        # Converter colunas categu00f3ricas para numu00e9ricas
        print("Convertendo colunas categu00f3ricas para numu00e9ricas...")
        df_processed = df.copy()
        
        # Converter Yes/No para 1/0
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"  Convertendo coluna categu00f3rica: {col}")
            # Mapear Yes/No para 1/0
            if set(df_processed[col].unique()) <= {'Yes', 'No'}:
                df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
            # Se houver outros valores, usar one-hot encoding
            else:
                # Criar dummies e anexar ao dataframe
                dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed.drop(columns=[col], inplace=True)
                print(f"    Aplicado one-hot encoding: {len(dummies.columns)} novas colunas")
        
        # Determinar as features (todas as colunas exceto as de target)
        target_columns = ['High_Death_Count', 'Total Deaths', 'No. Affected', 'Total Affected']
        feature_columns = [col for col in df_processed.columns if col not in target_columns]
        
        # Separar features e target
        X = df_processed[feature_columns].copy()
        y = df_processed[target_column].copy()
        
        # Verificar e tratar valores NaN nas features
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            print(f"Preenchendo {nan_count} valores NaN nas features com 0")
            X.fillna(0, inplace=True)
        
        # Dividir em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Dados ju00e1 estu00e3o normalizados, entu00e3o vamos pular a normalizau00e7u00e3o adicional
        print(f"Convertendo dados para o formato correto para TensorFlow")
        
        # Converter para float32 para compatibilidade com TensorFlow
        X_train_scaled = X_train.astype('float32')
        X_test_scaled = X_test.astype('float32')
        
        # Para classificau00e7u00e3o binu00e1ria e regressu00e3o
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')
        
        scaler_X = None
        scaler_y = None
        
        print(f"Dados preparados para {target_column}")
        print(f"  Features: {X_train_scaled.shape[1]}")
        print(f"  Treino: {X_train_scaled.shape[0]} amostras")
        print(f"  Teste: {X_test_scaled.shape[0]} amostras")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, scaler_y, feature_columns
    
    except Exception as e:
        print(f"Erro ao preparar dados para {target_column}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None

def create_mlp_model(input_dim, output_dim, problem_type):
    """
    Cria um modelo MLP (Perceptron de Mu00faltiplas Camadas) para classificau00e7u00e3o ou regressu00e3o.
    Versu00e3o simplificada para execuu00e7u00e3o mais ru00e1pida.
    
    Args:
        input_dim (int): Dimensu00e3o de entrada (nu00famero de features)
        output_dim (int): Dimensu00e3o de sau00edda (1 para regressu00e3o, 2 para classificau00e7u00e3o binu00e1ria)
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
    
    # Camada de sau00edda e compilau00e7u00e3o especu00edfica para o tipo de problema
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
    Versu00e3o simplificada para execuu00e7u00e3o mais ru00e1pida.
    
    Args:
        model: Modelo Keras compilado
        X_train, X_test, y_train, y_test: Dados de treino e teste
        model_name (str): Nome do modelo para salvar
        target_type (str): Tipo de alvo (binary_high_impact, mortality, affected)
    
    Returns:
        Histu00f3ria de treinamento, melhor modelo
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
    
    # Treinar o modelo com menos u00e9pocas para teste ru00e1pido
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Reduzido para execuu00e7u00e3o mais ru00e1pida
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Treinamento concluu00eddo em {training_time:.2f} segundos")
    
    # Carregar o melhor modelo
    best_model = load_model(model_path)
    print(f"Melhor modelo carregado de {model_path}")
    
    return history, best_model

def evaluate_model(model, X_test, y_test, scaler_y, problem_type, model_name, target_type):
    """
    Avalia um modelo neural e retorna as mu00e9tricas.
    
    Args:
        model: Modelo treinado
        X_test, y_test: Dados de teste
        scaler_y: Scaler para desfazer normalizau00e7u00e3o (None para classificau00e7u00e3o)
        problem_type: 'classification' ou 'regression'
        model_name, target_type: Informau00e7u00f5es do modelo
    
    Returns:
        Dictionary com mu00e9tricas de avaliau00e7u00e3o
    """
    print(f"\nAvaliando modelo {model_name} para {target_type}...")
    
    # Fazer previsu00f5es
    y_pred = model.predict(X_test)
    
    metrics = {}
    
    if problem_type == 'classification':
        # Transformar probabilidades em classes
        y_pred_class = (y_pred > 0.5).astype(int).flatten()
        
        # Calcular mu00e9tricas
        metrics['accuracy'] = accuracy_score(y_test, y_pred_class)
        metrics['precision'] = precision_score(y_test, y_pred_class)
        metrics['recall'] = recall_score(y_test, y_pred_class)
        metrics['f1'] = f1_score(y_test, y_pred_class)
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred.flatten())
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_class)
        
        # Exibir resultados
        print(f"Mu00e9tricas de classificau00e7u00e3o para {model_name} - {target_type}:")
        print(f"  Acuru00e1cia: {metrics['accuracy']:.4f}")
        print(f"  Precisu00e3o: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    else:  # regression
        # Dados ju00e1 estu00e3o normalizados, nu00e3o precisamos desnormalizar
        y_test_original = y_test
        y_pred = y_pred.flatten()
        
        # Calcular mu00e9tricas
        metrics['mse'] = mean_squared_error(y_test_original, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_test_original, y_pred)
        metrics['r2'] = r2_score(y_test_original, y_pred)
        
        # Exibir resultados
        print(f"Mu00e9tricas de regressu00e3o para {model_name} - {target_type}:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Ru00b2: {metrics['r2']:.4f}")
    
    return metrics

def main():
    """
    Funu00e7u00e3o principal para executar o teste ru00e1pido da rede neural MLP.
    """
    print("\n================================================================")
    print("TESTE Ru00c1PIDO DE REDE NEURAL - Versu00e3o otimizada para execuu00e7u00e3o ru00e1pida")
    print("================================================================\n")
    
    # Carregar os dados processados
    df = load_data()
    
    # Definir alvos para teste ru00e1pido usando os nomes corretos das colunas
    target_mapping = {
        'binary_high_impact': 'High_Death_Count',  # Coluna de classificau00e7u00e3o binu00e1ria
        'mortality': 'Total Deaths',             # Regressu00e3o para mortalidade
        'affected': 'No. Affected'               # Regressu00e3o para pessoas afetadas
    }
    
    targets = list(target_mapping.values())  # Usar os nomes das colunas reais do CSV
    
    for target_type in targets:
        print(f"\n================================================================")
        print(f"Desenvolvendo modelo para: {target_type}")
        print(f"================================================================\n")
        
        # Preparar os dados para o alvo especu00edfico
        problem_type = 'classification' if target_type == 'High_Death_Count' else 'regression'
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names = prepare_data_for_neural_networks(df, target_type)
        
        if X_train is None or X_test is None:
            print(f"Erro ao preparar dados para {target_type}. Pulando...")
            continue
        
        # Definir dimensu00f5es de entrada e sau00edda
        input_dim = X_train.shape[1]
        output_dim = 1  # Sempre 1 nesta versu00e3o simplificada
        
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
    print("Teste ru00e1pido de rede neural concluu00eddo com sucesso!")
    print("Os modelos foram salvos no diretu00f3rio de modelos.")
    print("================================================================\n")

if __name__ == '__main__':
    main()
