#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Desenvolvimento de redes neurais para previsão de impacto de desastres naturais.

Este script implementa diferentes arquiteturas de redes neurais para prever
impactos de desastres naturais, como classificação de alto impacto e regressão
para mortalidade e pessoas afetadas.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)

# Importar TensorFlow e Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input,
    LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.utils import to_categorical

# Definir sementes aleatórias para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# Definir diretórios para salvar modelos, relatórios e visualizações
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
REPORTS_DIR = os.path.join(BASE_DIR, 'results', 'reports')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots', 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'results', 'logs')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'results', 'processed_data')

# Garantir que os diretórios existam
def ensure_dir(directory):
    """Garante que um diretório existe, criando-o se necessário."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

for dir_path in [MODELS_DIR, REPORTS_DIR, PLOTS_DIR, LOGS_DIR]:
    ensure_dir(dir_path)

# Criar diretórios específicos para modelos de redes neurais
NN_MODELS_DIR = os.path.join(MODELS_DIR, 'neural_networks')
NN_PLOTS_DIR = os.path.join(PLOTS_DIR, 'neural_networks')
ensure_dir(NN_MODELS_DIR)

# Criar diretórios para cada tipo de modelo
for model_type in ['binary_high_impact', 'mortality', 'affected']:
    ensure_dir(os.path.join(NN_PLOTS_DIR, model_type))


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
    print(f"\nPreparando dados para modelo de {target_column}...")
    
    # Criar cópia para evitar warnings de visualização vs cópia
    data = df.copy()
    
    # Separar features e alvo
    if target_column == 'binary_high_impact':
        target = 'High_Death_Count'  # Classificação binária
    elif target_column == 'mortality':
        target = 'Total Deaths'  # Regressão
    elif target_column == 'affected':
        target = 'Total Affected'  # Regressão
    else:
        raise ValueError(f"Alvo desconhecido: {target_column}")
    
    # Verificar se a coluna alvo existe
    if target not in data.columns:
        print(f"Coluna alvo '{target}' não encontrada.")
        return None, None, None, None, None, None
    
    # Separar features (X) e alvo (y)
    y = data[target].copy()
    X = data.drop(columns=[target, 'Total Deaths', 'Total Affected', 
                          'High_Death_Count', 'Combined_Impact'], errors='ignore')
    
    # Remover colunas categóricas (object) e com valores ausentes
    # Redes neurais precisam de dados numéricos
    X = X.select_dtypes(exclude=['object'])
    X = X.dropna(axis=1)
    
    # Split em treino e teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalização das features (importante para redes neurais)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Se for problema de regressão, também normalizar o alvo
    scaler_y = None
    if target_column in ['mortality', 'affected']:
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    elif target_column == 'binary_high_impact':
        # Para classificação binária, converter para categorical
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
    
    print(f"Dados preparados: X_train={X_train_scaled.shape}, X_test={X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, scaler_y, X.columns.tolist()


def create_mlp_model(input_dim, output_dim, problem_type):
    """
    Cria um modelo MLP (Perceptron de Múltiplas Camadas) para classificação ou regressão.
    
    Args:
        input_dim (int): Dimensão de entrada (número de features)
        output_dim (int): Dimensão de saída (1 para regressão, 2 para classificação binária)
        problem_type (str): Tipo de problema ('classification' ou 'regression')
    
    Returns:
        Modelo Keras compilado
    """
    model = Sequential()
    
    # Camada de entrada
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Camadas ocultas
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Camada de saída
    if problem_type == 'classification':
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
    else:  # regression
        model.add(Dense(output_dim, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mae']
        )
    
    return model


def create_lstm_model(input_dim, output_dim, problem_type):
    """
    Cria um modelo LSTM para classificação ou regressão.
    
    Args:
        input_dim (int): Dimensão de entrada (número de features)
        output_dim (int): Dimensão de saída (1 para regressão, 2 para classificação binária)
        problem_type (str): Tipo de problema ('classification' ou 'regression')
    
    Returns:
        Modelo Keras compilado
    """
    # Para LSTM, precisamos reshapear os dados
    # Adicionamos uma dimensão de tempo (1) para cada amostra
    input_shape = (1, input_dim)
    
    model = Sequential()
    
    # Camada LSTM
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Camadas densas
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Camada de saída
    if problem_type == 'classification':
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
    else:  # regression
        model.add(Dense(output_dim, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mae']
        )
    
    return model


def create_cnn_model(input_dim, output_dim, problem_type):
    """
    Cria um modelo CNN 1D para classificação ou regressão.
    
    Args:
        input_dim (int): Dimensão de entrada (número de features)
        output_dim (int): Dimensão de saída (1 para regressão, 2 para classificação binária)
        problem_type (str): Tipo de problema ('classification' ou 'regression')
    
    Returns:
        Modelo Keras compilado
    """
    # Para CNN 1D, precisamos reshapear os dados
    # Adicionamos uma dimensão de tempo (1) para cada amostra
    input_shape = (input_dim, 1)
    
    model = Sequential()
    
    # Camadas convolucionais
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    # Flatten para conectar às camadas densas
    model.add(Flatten())
    
    # Camadas densas
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Camada de saída
    if problem_type == 'classification':
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
    else:  # regression
        model.add(Dense(output_dim, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mae']
        )
    
    return model
def reshape_data_for_model(X_train, X_test, model_type):
    """
    Reshape dos dados para o formato esperado pelos diferentes tipos de redes neurais.
    
    Args:
        X_train: Dados de treino
        X_test: Dados de teste
        model_type (str): Tipo de modelo ('mlp', 'lstm', 'cnn')
    
    Returns:
        X_train_reshaped, X_test_reshaped
    """
    if model_type == 'mlp':
        # MLP já usa o formato padrão (samples, features)
        return X_train, X_test
    
    elif model_type == 'lstm':
        # LSTM precisa do formato (samples, time_steps, features)
        X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        return X_train_reshaped, X_test_reshaped
    
    elif model_type == 'cnn':
        # CNN 1D precisa do formato (samples, features, channels)
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        return X_train_reshaped, X_test_reshaped
    
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")


def train_neural_network(model, X_train, X_test, y_train, y_test, model_name, target_type, model_type):
    """
    Treina uma rede neural e salva o melhor modelo.
    
    Args:
        model: Modelo Keras compilado
        X_train, X_test, y_train, y_test: Dados de treino e teste
        model_name (str): Nome do modelo para salvar
        target_type (str): Tipo de alvo (binary_high_impact, mortality, affected)
        model_type (str): Tipo de arquitetura (mlp, lstm, cnn)
    
    Returns:
        História de treinamento, melhor modelo
    """
    print(f"\nTreinando modelo {model_name} para {target_type} com arquitetura {model_type}...")
    
    # Diretório para salvar o melhor modelo
    model_save_path = os.path.join(NN_MODELS_DIR, f"{model_name}_{target_type}.h5")
    
    # Definir callbacks para monitorar o treinamento
    callbacks = [
        # Early stopping para evitar overfitting
        EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        # Salvar o melhor modelo
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Reduzir learning rate quando o treinamento estagna
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    # Treinar o modelo
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Máximo de épocas (early stopping pode interromper antes)
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_time
    print(f"Treinamento concluído em {train_time:.2f} segundos")
    
    # Carregar o melhor modelo salvo
    best_model = load_model(model_save_path)
    print(f"Melhor modelo carregado de {model_save_path}")
    
    return history, best_model


def evaluate_classification_neural_network(model, X_test, y_test, model_name, target_type):
    """
    Avalia um modelo de classificação neural e retorna as métricas.
    
    Args:
        model: Modelo treinado
        X_test, y_test: Dados de teste
        model_name (str): Nome do modelo
        target_type (str): Tipo de alvo
    
    Returns:
        Dictionary com métricas de avaliação
    """
    print(f"\nAvaliando modelo de classificação {model_name} para {target_type}...")
    
    # Fazer predições
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC AUC apenas para classificação binária
    roc_auc = roc_auc_score(y_true, y_pred_prob[:, 1]) if y_pred_prob.shape[1] == 2 else None
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Imprimir métricas
    print(f"  {model_name}:\n    Accuracy: {accuracy:.4f}\n    Precision: {precision:.4f}\n    Recall: {recall:.4f}\n    F1-Score: {f1:.4f}")
    if roc_auc is not None:
        roc_auc_str = f"{roc_auc:.4f}" if not pd.isna(roc_auc) else "N/A"
        print(f"    ROC AUC: {roc_auc_str}")
    
    # Retornar métricas em um dicionário
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }
    
    return metrics


def evaluate_regression_neural_network(model, X_test, y_test, scaler_y, model_name, target_type):
    """
    Avalia um modelo de regressão neural e retorna as métricas.
    
    Args:
        model: Modelo treinado
        X_test, y_test: Dados de teste
        scaler_y: Scaler usado para normalizar o alvo (para desfazer a normalização)
        model_name (str): Nome do modelo
        target_type (str): Tipo de alvo
    
    Returns:
        Dictionary com métricas de avaliação
    """
    print(f"\nAvaliando modelo de regressão {model_name} para {target_type}...")
    
    # Fazer predições
    y_pred = model.predict(X_test).flatten()
    
    # Desfazer a normalização
    if scaler_y is not None:
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_test_original = y_test
        y_pred_original = y_pred
    
    # Calcular métricas
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    
    # Imprimir métricas
    print(f"  {model_name}:\n    MSE: {mse:.4f}\n    RMSE: {rmse:.4f}\n    MAE: {mae:.4f}\n    R²: {r2:.4f}")
    
    # Retornar métricas em um dicionário
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_true': y_test_original,
        'y_pred': y_pred_original
    }
    
    return metrics


def generate_classification_plots(metrics, history, target_type, model_name):
    """
    Gera visualizações para modelos de classificação.
    
    Args:
        metrics (dict): Métricas de avaliação
        history: Histórico de treinamento
        target_type (str): Tipo de alvo
        model_name (str): Nome do modelo
    """
    plot_dir = os.path.join(NN_PLOTS_DIR, target_type)
    ensure_dir(plot_dir)
    
    # Extrair dados do histórico
    history_dict = history.history
    
    # 1. Curvas de aprendizado (Loss e Accuracy)
    plt.figure(figsize=(12, 5))
    
    # Subplot para Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Subplot para Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_learning_curves.png'))
    plt.close()
    
    # 2. Matriz de confusão
    plt.figure(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    # 3. Probabilidades previstas
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['y_pred_prob'][:, 1], bins=20, alpha=0.7)
    plt.title(f'{model_name} - Predicted Probabilities Distribution')
    plt.xlabel('Probability of Class 1')
    plt.ylabel('Frequency')
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_prob_distribution.png'))
    plt.close()
    
    print(f"Gráficos de classificação salvos em {plot_dir}")


def generate_regression_plots(metrics, history, target_type, model_name):
    """
    Gera visualizações para modelos de regressão.
    
    Args:
        metrics (dict): Métricas de avaliação
        history: Histórico de treinamento
        target_type (str): Tipo de alvo
        model_name (str): Nome do modelo
    """
    plot_dir = os.path.join(NN_PLOTS_DIR, target_type)
    ensure_dir(plot_dir)
    
    # Extrair dados do histórico
    history_dict = history.history
    
    # 1. Curvas de aprendizado (Loss e MAE)
    plt.figure(figsize=(12, 5))
    
    # Subplot para Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Subplot para MAE
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['mae'], label='Training MAE')
    plt.plot(history_dict['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_learning_curves.png'))
    plt.close()
    
    # 2. Valores reais vs. previstos
    plt.figure(figsize=(10, 8))
    plt.scatter(metrics['y_true'], metrics['y_pred'], alpha=0.5)
    plt.plot([metrics['y_true'].min(), metrics['y_true'].max()], 
             [metrics['y_true'].min(), metrics['y_true'].max()], 
             'k--', lw=2)
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_actual_vs_predicted.png'))
    plt.close()
    
    # 3. Distribuição dos erros
    errors = metrics['y_true'] - metrics['y_pred']
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.title(f'{model_name} - Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_error_distribution.png'))
    plt.close()
    
    print(f"Gráficos de regressão salvos em {plot_dir}")


def generate_model_report(model_name, target_type, metrics, feature_names, problem_type):
    """
    Gera um relatório em formato Markdown para o modelo.
    
    Args:
        model_name (str): Nome do modelo
        target_type (str): Tipo de alvo
        metrics (dict): Métricas de avaliação
        feature_names (list): Nomes das features usadas
        problem_type (str): Tipo de problema ('classification' ou 'regression')
    """
    report_path = os.path.join(REPORTS_DIR, f'neural_network_{target_type}_report.md')
    
    # Preparar conteúdo do relatório
    report_content = f"# Relatório de Modelo Neural - {target_type}\n\n"
    report_content += f"## Modelo: {model_name}\n\n"
    
    # Adicionar métricas
    report_content += "## Métricas de Avaliação\n\n"
    
    if problem_type == 'classification':
        report_content += f"- **Acurácia**: {metrics['accuracy']:.4f}\n"
        report_content += f"- **Precisão**: {metrics['precision']:.4f}\n"
        report_content += f"- **Recall**: {metrics['recall']:.4f}\n"
        report_content += f"- **F1-Score**: {metrics['f1']:.4f}\n"
        
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            roc_auc_str = f"{metrics['roc_auc']:.4f}" if not pd.isna(metrics['roc_auc']) else "N/A"
            report_content += f"- **ROC AUC**: {roc_auc_str}\n"
    else:  # regression
        report_content += f"- **MSE**: {metrics['mse']:.4f}\n"
        report_content += f"- **RMSE**: {metrics['rmse']:.4f}\n"
        report_content += f"- **MAE**: {metrics['mae']:.4f}\n"
        report_content += f"- **R²**: {metrics['r2']:.4f}\n"
    
    # Adicionar informações sobre as features
    report_content += "\n## Features Utilizadas\n\n"
    for i, feature in enumerate(feature_names, 1):
        report_content += f"{i}. {feature}\n"
    
    # Adicionar links para as visualizações
    report_content += "\n## Visualizações\n\n"
    plot_dir = os.path.join(NN_PLOTS_DIR, target_type)
    
    if problem_type == 'classification':
        report_content += f"- [Curvas de Aprendizado]({os.path.relpath(os.path.join(plot_dir, f'{model_name}_learning_curves.png'), BASE_DIR)})\n"
        report_content += f"- [Matriz de Confusão]({os.path.relpath(os.path.join(plot_dir, f'{model_name}_confusion_matrix.png'), BASE_DIR)})\n"
        report_content += f"- [Distribuição de Probabilidades]({os.path.relpath(os.path.join(plot_dir, f'{model_name}_prob_distribution.png'), BASE_DIR)})\n"
    else:  # regression
        report_content += f"- [Curvas de Aprendizado]({os.path.relpath(os.path.join(plot_dir, f'{model_name}_learning_curves.png'), BASE_DIR)})\n"
        report_content += f"- [Valores Reais vs. Previstos]({os.path.relpath(os.path.join(plot_dir, f'{model_name}_actual_vs_predicted.png'), BASE_DIR)})\n"
        report_content += f"- [Distribuição de Erros]({os.path.relpath(os.path.join(plot_dir, f'{model_name}_error_distribution.png'), BASE_DIR)})\n"
    
    # Adicionar conclusão
    report_content += "\n## Conclusão\n\n"
    
    if problem_type == 'classification':
        if metrics['accuracy'] > 0.7:
            conclusion = f"O modelo {model_name} apresentou um bom desempenho na classificação de {target_type}, com uma acurácia de {metrics['accuracy']:.4f} e F1-Score de {metrics['f1']:.4f}."
        else:
            conclusion = f"O modelo {model_name} apresentou um desempenho moderado na classificação de {target_type}, com uma acurácia de {metrics['accuracy']:.4f} e F1-Score de {metrics['f1']:.4f}. Pode ser necessário ajustar a arquitetura ou hiperparâmetros para melhorar o desempenho."
    else:  # regression
        if metrics['r2'] > 0.7:
            conclusion = f"O modelo {model_name} apresentou um bom desempenho na regressão de {target_type}, com um R² de {metrics['r2']:.4f} e RMSE de {metrics['rmse']:.4f}."
        else:
            conclusion = f"O modelo {model_name} apresentou um desempenho moderado na regressão de {target_type}, com um R² de {metrics['r2']:.4f} e RMSE de {metrics['rmse']:.4f}. Pode ser necessário ajustar a arquitetura ou hiperparâmetros para melhorar o desempenho."
    
    report_content += conclusion
    
    # Escrever o relatório
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Relatório salvo em {report_path}")
def main():
    """
    Função principal que orquestra todo o fluxo de desenvolvimento de redes neurais:
    - Carregamento de dados
    - Preparação dos dados
    - Treinamento de diferentes arquiteturas de redes neurais
    - Avaliação dos modelos
    - Geração de relatórios e visualizações
    """
    print("\n================================================================")
    print("Iniciando desenvolvimento de redes neurais para previsão de impactos de desastres naturais")
    print("================================================================\n")
    
    # Passo 1: Carregar os dados processados
    df = load_data()
    
    # Lista de arquiteturas a serem testadas
    architectures = ['mlp', 'lstm', 'cnn']
    
    # Lista de alvos para previsão
    targets = ['binary_high_impact', 'mortality', 'affected']
    
    # Dicionário para armazenar os melhores modelos e métricas
    best_models = {}
    
    # Para cada tipo de alvo, treinar e avaliar os modelos
    for target_type in targets:
        print(f"\n================================================================")
        print(f"Desenvolvendo modelos para: {target_type}")
        print(f"================================================================\n")
        
        # Preparar os dados para o alvo específico
        problem_type = 'classification' if target_type == 'binary_high_impact' else 'regression'
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names = prepare_data_for_neural_networks(df, target_type)
        
        if X_train is None or X_test is None:
            print(f"Erro ao preparar dados para {target_type}. Pulando...")
            continue
        
        # Armazenar as métricas de cada arquitetura
        metrics_by_arch = {}
        
        # Treinar e avaliar cada arquitetura
        for arch in architectures:
            # Definir dimensões de entrada e saída
            input_dim = X_train.shape[1]
            output_dim = 2 if problem_type == 'classification' else 1
            
            # Criar o modelo apropriado
            if arch == 'mlp':
                model = create_mlp_model(input_dim, output_dim, problem_type)
            elif arch == 'lstm':
                model = create_lstm_model(input_dim, output_dim, problem_type)
            elif arch == 'cnn':
                model = create_cnn_model(input_dim, output_dim, problem_type)
            
            # Reshapear os dados conforme necessário para cada arquitetura
            X_train_reshaped, X_test_reshaped = reshape_data_for_model(X_train, X_test, arch)
            
            # Treinar o modelo
            history, trained_model = train_neural_network(
                model, X_train_reshaped, X_test_reshaped, y_train, y_test, 
                arch, target_type, arch
            )
            
            # Avaliar o modelo
            if problem_type == 'classification':
                metrics = evaluate_classification_neural_network(
                    trained_model, X_test_reshaped, y_test, arch, target_type
                )
                generate_classification_plots(metrics, history, target_type, arch)
            else:  # regression
                metrics = evaluate_regression_neural_network(
                    trained_model, X_test_reshaped, y_test, scaler_y, arch, target_type
                )
                generate_regression_plots(metrics, history, target_type, arch)
            
            # Armazenar métricas para comparação
            metrics_by_arch[arch] = metrics
            
            # Gerar relatório para o modelo
            generate_model_report(arch, target_type, metrics, feature_names, problem_type)
        
        # Determinar o melhor modelo para este alvo
        if problem_type == 'classification':
            # Para classificação, usar F1-score como métrica principal
            best_arch = max(metrics_by_arch.items(), key=lambda x: x[1]['f1'])[0]
        else:  # regression
            # Para regressão, usar R² como métrica principal
            best_arch = max(metrics_by_arch.items(), key=lambda x: x[1]['r2'])[0]
        
        print(f"\nMelhor arquitetura para {target_type}: {best_arch}")
        best_models[target_type] = best_arch
    
    # Resumo final
    print("\n================================================================")
    print("Resumo dos melhores modelos por tarefa:")
    print("================================================================")
    
    for target, arch in best_models.items():
        print(f"- {target}: {arch}")
    
    print("\nDesenvolvimento de redes neurais concluído com sucesso!")
    print("Os modelos foram salvos, avaliados e relatórios foram gerados.")
    print("Verifique os diretórios de modelos, relatórios e visualizações para mais detalhes.")


if __name__ == '__main__':
    main()
