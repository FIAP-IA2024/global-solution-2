#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Desenvolvimento de modelos de machine learning para previsão de impacto de desastres naturais.

Este script realiza o treinamento, avaliação e seleção de múltiplos modelos de
classificação e regressão para prever impactos de desastres naturais.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from tabulate import tabulate
from sklearn.model_selection import train_test_split

# Classificação
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Regressão
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Adicionar o diretório raiz ao path para permitir importações relativas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar funções auxiliares
from src.utils.helpers import create_directory

# Definir diretórios para salvar resultados
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots', 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'results', 'reports')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'results', 'processed_data')

# Garantir que os diretórios existam
create_directory(MODELS_DIR)
create_directory(PLOTS_DIR)
create_directory(REPORTS_DIR)

def load_processed_data():
    """Carrega os dados processados do arquivo CSV.
    
    Returns:
        DataFrame: Dados processados.
    """
    try:
        # Tentar carregar do diretório results/processed_data
        processed_data_file = os.path.join(PROCESSED_DATA_DIR, 'processed_disasters_data.csv')
        df = pd.read_csv(processed_data_file)
        print(f"Dados carregados com sucesso de {processed_data_file}")
        print(f"Forma dos dados: {df.shape}")
        print(f"Colunas: {df.columns.tolist()[:5]}...")
        return df
    except FileNotFoundError:
        # Se não encontrar, tentar carregar do diretório data
        try:
            data_file = os.path.join(BASE_DIR, 'data', 'processed_disasters_data.csv')
            df = pd.read_csv(data_file)
            print(f"Dados carregados com sucesso de {data_file}")
            print(f"Forma dos dados: {df.shape}")
            print(f"Colunas: {df.columns.tolist()[:5]}...")
            return df
        except FileNotFoundError:
            print("ERRO: Arquivo de dados processados não encontrado.")
            print("Execute primeiro o script de pré-processamento de dados.")
            return None


def prepare_data_for_modeling(df, target_type='binary_high_impact', test_size=0.2, random_state=42):
    """Prepara os dados para modelagem, separando features e target, e dividindo em treino e teste.
    
    Args:
        df (DataFrame): Dados processados
        target_type (str): Tipo de alvo a ser previsto:
            - 'binary_high_impact': Classificação binária para alto impacto
            - 'mortality': Regressão para número de mortes
            - 'affected': Regressão para número de afetados
        test_size (float): Proporção do conjunto de teste
        random_state (int): Semente aleatória para reprodutibilidade
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, target_type)
    """
    if df is None:
        return None, None, None, None, target_type
    
    print(f"\nPreparando dados para modelagem: {target_type}")
    
    # Definir colunas alvo com base no tipo
    if target_type == 'binary_high_impact':
        target_column = 'High_Death_Count'
    elif target_type == 'mortality':
        target_column = 'Total Deaths'
    elif target_type == 'affected':
        target_column = 'Total Affected'
    else:
        print(f"ERRO: Tipo de alvo '{target_type}' não reconhecido.")
        return None, None, None, None, target_type
    
    # Verificar se a coluna alvo existe
    if target_column not in df.columns:
        print(f"ERRO: Coluna alvo '{target_column}' não encontrada nos dados.")
        return None, None, None, None, target_type
    
    # Separar features e target
    X = df.drop(columns=['Total Deaths', 'Total Affected', 'High_Death_Count', 'Combined_Impact'], errors='ignore')
    y = df[target_column]
    
    # Remover colunas categóricas e com valores NaN
    X = X.select_dtypes(exclude=['object'])
    X = X.dropna(axis=1)
    
    print(f"Removidas colunas categóricas e com valores ausentes. Restaram {X.shape[1]} features.")
    
    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"Dados divididos em:\n  - Treino: {X_train.shape[0]} amostras ({X_train.shape[1]} features)\n  - Teste: {X_test.shape[0]} amostras ({X_test.shape[1]} features)")
    
    return X_train, X_test, y_train, y_test, target_type


def train_classification_models(X_train, y_train, random_state=42):
    """Treina múltiplos modelos de classificação.
    
    Args:
        X_train (DataFrame): Features de treinamento
        y_train (Series): Alvo de treinamento
        random_state (int): Semente aleatória para reprodutibilidade
        
    Returns:
        dict: Modelos treinados
    """
    if X_train is None or y_train is None:
        return None
    
    print("\nTreinando modelos de classificação...")
    
    # Inicializar modelos de classificação
    models = {
        'logistic_regression': {
            'model': LogisticRegression(max_iter=1000, random_state=random_state)
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(random_state=random_state)
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=random_state)
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=random_state)
        },
        'svm': {
            'model': SVC(probability=True, random_state=random_state)
        },
        'knn': {
            'model': KNeighborsClassifier()
        }
    }
    
    # Treinar cada modelo e medir o tempo
    for name, model_info in models.items():
        model = model_info['model']
        print(f"  Treinando {name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        model_info['train_time'] = train_time
        print(f"    Tempo de treinamento: {train_time:.2f} segundos")
    
    return models


def train_regression_models(X_train, y_train, random_state=42):
    """Treina múltiplos modelos de regressão.
    
    Args:
        X_train (DataFrame): Features de treinamento
        y_train (Series): Alvo de treinamento
        random_state (int): Semente aleatória para reprodutibilidade
        
    Returns:
        dict: Modelos treinados
    """
    if X_train is None or y_train is None:
        return None
    
    print("\nTreinando modelos de regressão...")
    
    # Inicializar modelos de regressão
    models = {
        'linear_regression': {
            'model': LinearRegression()
        },
        'ridge': {
            'model': Ridge(random_state=random_state)
        },
        'lasso': {
            'model': Lasso(random_state=random_state)
        },
        'random_forest_regressor': {
            'model': RandomForestRegressor(random_state=random_state)
        },
        'gradient_boosting_regressor': {
            'model': GradientBoostingRegressor(random_state=random_state)
        },
        'svr': {
            'model': SVR()
        }
    }
    
    # Treinar cada modelo e medir o tempo
    for name, model_info in models.items():
        model = model_info['model']
        print(f"  Treinando {name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        model_info['train_time'] = train_time
        print(f"    Tempo de treinamento: {train_time:.2f} segundos")
    
    return models


def evaluate_classification_models(models, X_test, y_test):
    """Avalia os modelos de classificação treinados.
    
    Args:
        models (dict): Modelos treinados
        X_test (DataFrame): Features de teste
        y_test (Series): Alvo de teste
        
    Returns:
        dict: Resultados da avaliação
    """
    if models is None or X_test is None or y_test is None:
        return None
    
    print("\nAvaliando modelos de classificação...")
    
    results = {}
    
    for name, model_info in models.items():
        model = model_info['model']
        
        # Previsões
        y_pred = model.predict(X_test)
        
        # Métricas de avaliação
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC AUC se o modelo pode estimar probabilidades
        roc_auc = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            except:
                pass
        
        # Armazenar resultados
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'train_time': model_info['train_time'],
            'model': model
        }
        
        roc_auc_str = f"{roc_auc:.4f}" if not pd.isna(roc_auc) else "N/A"
        print(f"  {name}:\n    Acurácia: {accuracy:.4f}\n    Precisão: {precision:.4f}\n    Recall: {recall:.4f}\n    F1-Score: {f1:.4f}\n    ROC AUC: {roc_auc_str}")
    
    return results


def evaluate_regression_models(models, X_test, y_test):
    """Avalia os modelos de regressão treinados.
    
    Args:
        models (dict): Modelos treinados
        X_test (DataFrame): Features de teste
        y_test (Series): Alvo de teste
        
    Returns:
        dict: Resultados da avaliação
    """
    if models is None or X_test is None or y_test is None:
        return None
    
    print("\nAvaliando modelos de regressão...")
    
    results = {}
    
    for name, model_info in models.items():
        model = model_info['model']

        # Previsões
        y_pred = model.predict(X_test)

        # Métricas de avaliação
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Armazenar resultados
        results[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'train_time': model_info['train_time'],
            'model': model
        }
        
        print(f"  {name}:\n    MSE: {mse:.4f}\n    RMSE: {rmse:.4f}\n    MAE: {mae:.4f}\n    R²: {r2:.4f}")
    
    return results


def save_best_model(results, target_type):
    """Salva o melhor modelo com base nas métricas.
    
    Args:
        results (dict): Resultados da avaliação
        target_type (str): Tipo de alvo previsto
        
    Returns:
        str: Nome do melhor modelo salvo
    """
    if results is None:
        return None
    
    print(f"\nSalvando o melhor modelo para {target_type}...")
    
    # Selecionar o melhor modelo com base na métrica relevante
    if target_type == 'binary_high_impact':
        # Para classificação, escolher com base no F1-score
        best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
        metric_name = 'f1_score'
        metric_value = results[best_model_name]['f1_score']
    else:
        # Para regressão, escolher com base no R²
        best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
        metric_name = 'r2'
        metric_value = results[best_model_name]['r2']
    
    # Preparar o modelo para salvar
    best_model = results[best_model_name]['model']
    model_filename = f"{best_model_name}_{target_type}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    # Salvar o modelo usando pickle
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"  Melhor modelo: {best_model_name}")
    print(f"  {metric_name.upper()}: {metric_value:.4f}")
    print(f"  Modelo salvo em: {model_path}")
    
    return best_model_name


def generate_model_report(results, target_type, best_model_name=None):
    """Gera um relatório detalhado dos resultados dos modelos.
    
    Args:
        results (dict): Resultados da avaliação
        target_type (str): Tipo de alvo previsto
        best_model_name (str, optional): Nome do melhor modelo
        
    Returns:
        str: Caminho para o arquivo de relatório gerado
    """
    if results is None:
        return None
    
    print(f"\nGerando relatório para modelos de {target_type}...")
    
    # Definir o título do relatório
    if target_type == 'binary_high_impact':
        report_title = "Relatório de Modelos de Classificação para Impacto Alto"
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'train_time']
        report_filename = "classification_models_report.md"
    elif target_type == 'mortality':
        report_title = "Relatório de Modelos de Regressão para Mortalidade"
        metrics = ['mse', 'rmse', 'mae', 'r2', 'train_time']
        report_filename = "regression_mortality_models_report.md"
    elif target_type == 'affected':
        report_title = "Relatório de Modelos de Regressão para Pessoas Afetadas"
        metrics = ['mse', 'rmse', 'mae', 'r2', 'train_time']
        report_filename = "regression_affected_models_report.md"
    else:
        print(f"ERRO: Tipo de alvo '{target_type}' não reconhecido.")
        return None
    
    # Gerar timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Iniciar o relatório
    report = f"# {report_title}\n\n"
    report += f"**Data de geração:** {timestamp}\n\n"
    report += "## Resumo dos Resultados\n\n"
    
    # Preparar tabela de comparação
    table_data = []
    headers = ['Modelo'] + [m.upper() for m in metrics]
    
    for name, result in results.items():
        row = [name]
        for metric in metrics:
            if metric in result and result[metric] is not None:
                if metric == 'train_time':
                    row.append(f"{result[metric]:.2f}s")
                else:
                    row.append(f"{result[metric]:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Adicionar tabela ao relatório
    report += tabulate(table_data, headers=headers, tablefmt="pipe") + "\n\n"
    
    # Destacar o melhor modelo
    if best_model_name:
        report += f"## Melhor Modelo\n\n"
        report += f"O melhor modelo para {target_type} é **{best_model_name}**"  
        
        if target_type == 'binary_high_impact':
            metric_name = "F1-Score"
            metric_value = results[best_model_name]['f1_score']
        else:
            metric_name = "R²"
            metric_value = results[best_model_name]['r2']
            
        report += f" com {metric_name} de **{metric_value:.4f}**.\n\n"
    
    # Detalhes dos modelos
    report += "## Detalhes dos Modelos\n\n"
    
    for name, result in results.items():
        report += f"### {name}\n\n"
        
        if target_type == 'binary_high_impact':
            report += f"- **Acurácia:** {result['accuracy']:.4f}\n"
            report += f"- **Precisão:** {result['precision']:.4f}\n"
            report += f"- **Recall:** {result['recall']:.4f}\n"
            report += f"- **F1-Score:** {result['f1_score']:.4f}\n"
            if 'roc_auc' in result and result['roc_auc'] is not None:
                report += f"- **ROC AUC:** {result['roc_auc']:.4f}\n"
        else:
            report += f"- **MSE:** {result['mse']:.4f}\n"
            report += f"- **RMSE:** {result['rmse']:.4f}\n"
            report += f"- **MAE:** {result['mae']:.4f}\n"
            report += f"- **R²:** {result['r2']:.4f}\n"
        
        report += f"- **Tempo de Treinamento:** {result['train_time']:.2f} segundos\n\n"
    
    # Notas e conclusões
    report += "## Notas e Conclusões\n\n"
    report += "Este relatório foi gerado automaticamente pelo script de desenvolvimento de modelos. "
    report += "As métricas apresentadas são baseadas no conjunto de teste. "
    report += "O melhor modelo foi salvo no diretório de modelos para uso futuro.\n"
    
    # Salvar o relatório
    report_path = os.path.join(REPORTS_DIR, report_filename)
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"  Relatório salvo em: {report_path}")
    
    return report_path


def generate_model_visualizations(results, X_test, y_test, target_type):
    """Gera visualizações para os modelos treinados.
    
    Args:
        results (dict): Resultados da avaliação
        X_test (DataFrame): Features de teste
        y_test (Series): Alvo de teste
        target_type (str): Tipo de alvo previsto
        
    Returns:
        list: Caminhos para os arquivos de visualização gerados
    """
    if results is None or X_test is None or y_test is None:
        return None
    
    print(f"\nGerando visualizações para modelos de {target_type}...")
    
    # Preparar diretório para salvar as visualizações
    vis_dir = os.path.join(PLOTS_DIR, target_type)
    create_directory(vis_dir)
    
    plot_paths = []
    
    # Definir o que visualizar com base no tipo de alvo
    if target_type == 'binary_high_impact':
        # Matriz de confusão para cada modelo
        for name, result in results.items():
            model = result['model']
            y_pred = model.predict(X_test)
            
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Baixo Impacto', 'Alto Impacto'],
                       yticklabels=['Baixo Impacto', 'Alto Impacto'])
            plt.xlabel('Previsto')
            plt.ylabel('Real')
            plt.title(f'Matriz de Confusão - {name}')
            
            # Salvar figura
            plot_path = os.path.join(vis_dir, f"{name}_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            plot_paths.append(plot_path)
            print(f"  Matriz de confusão para {name} salva em: {plot_path}")
            
        # Comparação de métricas entre modelos
        plt.figure(figsize=(12, 8))
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        
        x = np.arange(len(metrics))
        width = 0.1
        offsets = np.linspace(-(len(results)-1)/2*width, (len(results)-1)/2*width, len(results))
        
        for i, (name, result) in enumerate(results.items()):
            values = [result[m] for m in metrics]
            plt.bar(x + offsets[i], values, width, label=name)
        
        plt.xticks(x, metric_names)
        plt.ylim(0, 1.0)
        plt.xlabel('Métrica')
        plt.ylabel('Valor')
        plt.title('Comparação de Métricas entre Modelos de Classificação')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Salvar figura
        plot_path = os.path.join(vis_dir, "model_metrics_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        print(f"  Comparação de métricas salva em: {plot_path}")
        
    else:  # Para modelos de regressão
        # Previsão vs. Real para cada modelo
        for name, result in results.items():
            model = result['model']
            y_pred = model.predict(X_test)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            
            # Linha de referência ideal
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Valor Real')
            plt.ylabel('Valor Previsto')
            plt.title(f'Previsto vs. Real - {name} (R² = {result["r2"]:.4f})')
            
            # Salvar figura
            plot_path = os.path.join(vis_dir, f"{name}_predicted_vs_actual.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            plot_paths.append(plot_path)
            print(f"  Gráfico de previsão vs. real para {name} salvo em: {plot_path}")
        
        # Comparação de métricas de erro entre modelos
        plt.figure(figsize=(10, 6))
        metrics = ['rmse', 'mae']
        metric_names = ['RMSE', 'MAE']
        
        x = np.arange(len(metrics))
        width = 0.1
        offsets = np.linspace(-(len(results)-1)/2*width, (len(results)-1)/2*width, len(results))
        
        for i, (name, result) in enumerate(results.items()):
            values = [result[m] for m in metrics]
            plt.bar(x + offsets[i], values, width, label=name)
        
        plt.xticks(x, metric_names)
        plt.xlabel('Métrica de Erro')
        plt.ylabel('Valor')
        plt.title(f'Comparação de Métricas de Erro entre Modelos de Regressão ({target_type})')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Salvar figura
        plot_path = os.path.join(vis_dir, "model_error_metrics_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        print(f"  Comparação de métricas de erro salva em: {plot_path}")
        
        # Comparação de R² entre modelos
        plt.figure(figsize=(10, 6))
        r2_values = [result['r2'] for name, result in results.items()]
        model_names = list(results.keys())
        
        plt.bar(model_names, r2_values)
        plt.ylim(0, 1.0)
        plt.xlabel('Modelo')
        plt.ylabel('R²')
        plt.title(f'Comparação de R² entre Modelos de Regressão ({target_type})')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Salvar figura
        plot_path = os.path.join(vis_dir, "model_r2_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        print(f"  Comparação de R² salva em: {plot_path}")
    
    return plot_paths

def main():
    """Função principal para executar o fluxo completo de desenvolvimento de modelos."""
    print("\n=== DESENVOLVIMENTO DE MODELOS DE MACHINE LEARNING PARA PREVISÃO DE DESASTRES NATURAIS ===")
    print("\nIniciando processamento...")
    
    # Carregar dados processados
    df = load_processed_data()
    if df is None:
        print("\nERRO: Não foi possível carregar os dados. Encerrando execução.")
        return
    
    # Desenvolver modelo de classificação para impacto alto
    print("\n===== MODELO DE CLASSIFICAÇÃO PARA IMPACTO ALTO =====")
    X_train, X_test, y_train, y_test, target_type = prepare_data_for_modeling(df, target_type='binary_high_impact')
    classification_models = train_classification_models(X_train, y_train)
    classification_results = evaluate_classification_models(classification_models, X_test, y_test)
    best_classification_model = save_best_model(classification_results, target_type)
    classification_report_path = generate_model_report(classification_results, target_type, best_classification_model)
    classification_plot_paths = generate_model_visualizations(classification_results, X_test, y_test, target_type)
    
    # Desenvolver modelo de regressão para mortalidade
    print("\n===== MODELO DE REGRESSÃO PARA MORTALIDADE =====")
    X_train, X_test, y_train, y_test, target_type = prepare_data_for_modeling(df, target_type='mortality')
    regression_models_mortality = train_regression_models(X_train, y_train)
    regression_results_mortality = evaluate_regression_models(regression_models_mortality, X_test, y_test)
    best_regression_model_mortality = save_best_model(regression_results_mortality, target_type)
    regression_mortality_report_path = generate_model_report(regression_results_mortality, target_type, best_regression_model_mortality)
    regression_mortality_plot_paths = generate_model_visualizations(regression_results_mortality, X_test, y_test, target_type)
    
    # Desenvolver modelo de regressão para pessoas afetadas
    print("\n===== MODELO DE REGRESSÃO PARA PESSOAS AFETADAS =====")
    X_train, X_test, y_train, y_test, target_type = prepare_data_for_modeling(df, target_type='affected')
    regression_models_affected = train_regression_models(X_train, y_train)
    regression_results_affected = evaluate_regression_models(regression_models_affected, X_test, y_test)
    best_regression_model_affected = save_best_model(regression_results_affected, target_type)
    regression_affected_report_path = generate_model_report(regression_results_affected, target_type, best_regression_model_affected)
    regression_affected_plot_paths = generate_model_visualizations(regression_results_affected, X_test, y_test, target_type)
    
    # Resumo final
    print("\n===== RESUMO FINAL =====")
    print("\nModelos treinados e avaliados com sucesso!")
    print("Os melhores modelos foram salvos e estão prontos para uso.")
    print("\nMelhores modelos:\n")
    
    if best_classification_model:
        print(f"  - Classificação (Impacto Alto): {best_classification_model}")
        print(f"    F1-Score: {classification_results[best_classification_model]['f1_score']:.4f}")
    
    if best_regression_model_mortality:
        print(f"  - Regressão (Mortalidade): {best_regression_model_mortality}")
        print(f"    R²: {regression_results_mortality[best_regression_model_mortality]['r2']:.4f}")
    
    if best_regression_model_affected:
        print(f"  - Regressão (Pessoas Afetadas): {best_regression_model_affected}")
        print(f"    R²: {regression_results_affected[best_regression_model_affected]['r2']:.4f}")
    
    print("\nRelatórios e visualizações disponíveis em:\n")
    print(f"  - Relatórios: {REPORTS_DIR}")
    print(f"  - Visualizações: {PLOTS_DIR}")
    print(f"  - Modelos salvos: {MODELS_DIR}")
    
    print("\nProcesso de desenvolvimento de modelos concluído com sucesso!\n")


# Executar se o script for executado diretamente (não importado)
if __name__ == '__main__':
    main()