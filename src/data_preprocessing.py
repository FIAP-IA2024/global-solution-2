#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing for Natural Disasters Dataset

Este script realiza o pré-processamento dos dados do dataset de desastres naturais,
focando nos tipos de desastre prioritários (Storm, Flood, Earthquake) identificados
na análise exploratória. Inclui limpeza de dados, tratamento de valores ausentes,
normalização e engenharia de features.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Adicionar o caminho do projeto ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import helper functions
from utils.helpers import create_output_dir, save_plot

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Define paths
DATASET_PATH = '/Users/gabrielribeiro/www/fiap/global-solution-2/dataset.xlsx'
PROCESSED_DIR = 'results/processed_data'
PLOTS_DIR = 'results/plots/preprocessing'

# Criar diretórios se não existirem
create_output_dir(PROCESSED_DIR)
create_output_dir(PLOTS_DIR)

# Tipos de desastre prioritários identificados na análise exploratória
PRIORITY_DISASTERS = ['Storm', 'Flood', 'Earthquake']


def load_data(file_path=DATASET_PATH):
    """Carregar o dataset a partir do arquivo Excel."""
    print(f"Carregando dados de {file_path}...")
    df = pd.read_excel(file_path)
    print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df


def filter_priority_disasters(df, disaster_types=PRIORITY_DISASTERS):
    """Filtrar o dataset para incluir apenas os tipos de desastre prioritários."""
    print(f"\nFiltrando dataset para tipos de desastre prioritários: {disaster_types}")
    filtered_df = df[df['Disaster Type'].isin(disaster_types)].copy()
    print(f"Dataset filtrado: {filtered_df.shape[0]} linhas e {filtered_df.shape[1]} colunas.")
    
    # Salvar estatísticas dos tipos de desastre prioritários
    disaster_counts = filtered_df['Disaster Type'].value_counts()
    print("\nContagem por tipo de desastre:")
    print(disaster_counts)
    
    # Plotar distribuição dos tipos de desastre após filtragem
    plt.figure(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='Disaster Type', order=disaster_counts.index)
    plt.title('Distribuição dos Tipos de Desastre Prioritários', fontsize=15)
    plt.xlabel('Tipo de Desastre')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(plt.gcf(), "priority_disaster_distribution.png", PLOTS_DIR)
    plt.close()
    
    return filtered_df


def analyze_missing_values(df):
    """Analisar e visualizar valores ausentes no dataset."""
    print("\nAnalisando valores ausentes...")
    
    # Calcular valores ausentes por coluna
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Valores Ausentes': missing_values,
        'Porcentagem': missing_percent
    }).sort_values('Porcentagem', ascending=False)
    
    # Filtrar colunas com valores ausentes
    missing_df = missing_df[missing_df['Valores Ausentes'] > 0]
    
    print("\nColunas com valores ausentes:")
    print(missing_df)
    
    # Plotar valores ausentes
    plt.figure(figsize=(14, 8))
    sns.barplot(x=missing_df.index, y=missing_df['Porcentagem'])
    plt.title('Porcentagem de Valores Ausentes por Coluna', fontsize=15)
    plt.xlabel('Colunas')
    plt.ylabel('Porcentagem de Valores Ausentes (%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_plot(plt.gcf(), "missing_values_percent.png", PLOTS_DIR)
    plt.close()
    
    return missing_df


def create_date_features(df):
    """Criar features temporais a partir das colunas de data."""
    print("\nCriando features temporais...")
    
    # Criar cópia para não modificar o dataframe original
    df_temp = df.copy()
    
    # Preencher valores ausentes em Start Month e Start Day com valores padrão (1)
    df_temp['Start Month'] = df_temp['Start Month'].fillna(1)
    df_temp['Start Day'] = df_temp['Start Day'].fillna(1)
    
    # Converter para inteiros (necessário para criar data)
    df_temp['Start Year'] = df_temp['Start Year'].astype(int)
    df_temp['Start Month'] = df_temp['Start Month'].astype(int)
    df_temp['Start Day'] = df_temp['Start Day'].astype(int)
    
    # Criar coluna de data de início
    df_temp['Start Date'] = pd.to_datetime(
        df_temp[['Start Year', 'Start Month', 'Start Day']].astype(str).agg('-'.join, axis=1),
        errors='coerce'
    )
    
    # Extrair componentes temporais
    df_temp['Start_Year'] = df_temp['Start Date'].dt.year
    df_temp['Start_Month'] = df_temp['Start Date'].dt.month
    df_temp['Start_Day'] = df_temp['Start Date'].dt.day
    df_temp['Start_DayOfWeek'] = df_temp['Start Date'].dt.dayofweek
    df_temp['Start_Quarter'] = df_temp['Start Date'].dt.quarter
    df_temp['Start_DayOfYear'] = df_temp['Start Date'].dt.dayofyear
    df_temp['Start_WeekOfYear'] = df_temp['Start Date'].dt.isocalendar().week
    
    # Se houver dados de End Date, criar features similares
    if 'End Month' in df_temp.columns and 'End Day' in df_temp.columns and 'End Year' in df_temp.columns:
        # Preencher valores ausentes
        df_temp['End Year'] = df_temp['End Year'].fillna(df_temp['Start Year'])
        df_temp['End Month'] = df_temp['End Month'].fillna(df_temp['Start Month'])
        df_temp['End Day'] = df_temp['End Day'].fillna(df_temp['Start Day'])
        
        # Converter para inteiros
        df_temp['End Year'] = df_temp['End Year'].astype(int)
        df_temp['End Month'] = df_temp['End Month'].astype(int)
        df_temp['End Day'] = df_temp['End Day'].astype(int)
        
        # Criar coluna de data de fim
        df_temp['End Date'] = pd.to_datetime(
            df_temp[['End Year', 'End Month', 'End Day']].astype(str).agg('-'.join, axis=1),
            errors='coerce'
        )
        
        # Calcular duração do desastre em dias
        df_temp['Disaster_Duration_Days'] = (df_temp['End Date'] - df_temp['Start Date']).dt.days
        
        # Substituir valores negativos por 1 (erro nos dados)
        df_temp['Disaster_Duration_Days'] = df_temp['Disaster_Duration_Days'].apply(lambda x: 1 if x < 0 else x)
    
    print("Features temporais criadas com sucesso.")
    
    return df_temp


def handle_missing_values(df, categorical_threshold=0.3, numerical_threshold=0.5):
    """Tratar valores ausentes no dataset com base em limiares definidos."""
    print("\nTratando valores ausentes...")
    
    # Identificar colunas categóricas e numéricas
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remover colunas com muitos valores ausentes
    missing_percent = df.isnull().mean()
    
    # Colunas categóricas com muitos valores ausentes
    cat_cols_to_drop = [col for col in categorical_cols if missing_percent[col] > categorical_threshold]
    
    # Colunas numéricas com muitos valores ausentes
    num_cols_to_drop = [col for col in numerical_cols if missing_percent[col] > numerical_threshold]
    
    cols_to_drop = cat_cols_to_drop + num_cols_to_drop
    
    print(f"Removendo {len(cols_to_drop)} colunas com muitos valores ausentes: {cols_to_drop}")
    df_cleaned = df.drop(columns=cols_to_drop)
    
    # Atualizar listas de colunas após remoção
    categorical_cols = [col for col in categorical_cols if col not in cat_cols_to_drop]
    numerical_cols = [col for col in numerical_cols if col not in num_cols_to_drop]
    
    # Preencher valores ausentes restantes
    # Para colunas categóricas: preencher com 'Unknown'
    for col in categorical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna('Unknown')
    
    # Para colunas numéricas: preencher com mediana
    for col in numerical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            median_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_value)
    
    print(f"Valores ausentes tratados. Dataset resultante: {df_cleaned.shape[0]} linhas e {df_cleaned.shape[1]} colunas.")
    
    return df_cleaned, categorical_cols, numerical_cols


def engineer_features(df):
    """Criar novas features a partir dos dados existentes."""
    print("\nCriando novas features...")
    
    # Criar cópia para não modificar o dataframe original
    df_new = df.copy()
    
    # 1. Razão entre mortes e total de afetados (índice de letalidade)
    if 'Total Deaths' in df_new.columns and 'Total Affected' in df_new.columns:
        df_new['Lethality_Index'] = df_new['Total Deaths'] / (df_new['Total Affected'] + 1)  # +1 para evitar divisão por zero
    
    # 2. Impacto normalizado (combinação de mortes, afetados e danos)
    if all(col in df_new.columns for col in ['Total Deaths', 'Total Affected', "Total Damage, Adjusted ('000 US$)"]):
        # Normalizar cada métrica de impacto para escala 0-1
        df_new['Normalized_Deaths'] = df_new['Total Deaths'] / df_new['Total Deaths'].max()
        df_new['Normalized_Affected'] = df_new['Total Affected'] / df_new['Total Affected'].max()
        df_new['Normalized_Damage'] = df_new["Total Damage, Adjusted ('000 US$)"] / df_new["Total Damage, Adjusted ('000 US$)"].max()
        
        # Calcular impacto combinado (média ponderada)
        df_new['Combined_Impact'] = (
            0.4 * df_new['Normalized_Deaths'] + 
            0.3 * df_new['Normalized_Affected'] + 
            0.3 * df_new['Normalized_Damage']
        )
    
    # 3. Flag para desastres com mortes acima da média
    if 'Total Deaths' in df_new.columns:
        deaths_mean = df_new['Total Deaths'].mean()
        df_new['High_Death_Count'] = df_new['Total Deaths'].apply(lambda x: 1 if x > deaths_mean else 0)
    
    # 4. Métricas por grupo geográfico (país, região)
    if 'Country' in df_new.columns and 'Total Deaths' in df_new.columns:
        # Média de mortes por país
        country_death_means = df_new.groupby('Country')['Total Deaths'].mean()
        df_new['Country_Avg_Deaths'] = df_new['Country'].map(country_death_means)
    
    if 'Region' in df_new.columns and 'Total Deaths' in df_new.columns:
        # Média de mortes por região
        region_death_means = df_new.groupby('Region')['Total Deaths'].mean()
        df_new['Region_Avg_Deaths'] = df_new['Region'].map(region_death_means)
    
    # 5. Variáveis cíclicas para mês e dia da semana (para capturar sazonalidade)
    if 'Start_Month' in df_new.columns:
        df_new['Month_Sin'] = np.sin(2 * np.pi * df_new['Start_Month'] / 12)
        df_new['Month_Cos'] = np.cos(2 * np.pi * df_new['Start_Month'] / 12)
    
    if 'Start_DayOfWeek' in df_new.columns:
        df_new['Day_Sin'] = np.sin(2 * np.pi * df_new['Start_DayOfWeek'] / 7)
        df_new['Day_Cos'] = np.cos(2 * np.pi * df_new['Start_DayOfWeek'] / 7)
    
    print(f"Novas features criadas. Dataset resultante: {df_new.shape[0]} linhas e {df_new.shape[1]} colunas.")
    
    return df_new


def normalize_numerical_features(df, numerical_cols):
    """Normalizar features numéricas para melhorar o desempenho dos modelos de ML."""
    print("\nNormalizando features numéricas...")
    
    # Remover colunas de identificação ou já normalizadas
    exclude_cols = [
        'DisNo.', 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day',
        'Normalized_Deaths', 'Normalized_Affected', 'Normalized_Damage', 'High_Death_Count'
    ]
    
    cols_to_normalize = [col for col in numerical_cols if col not in exclude_cols]
    
    if cols_to_normalize:
        # Criar um dataframe de cópia para não modificar o original
        df_normalized = df.copy()
        
        # Normalizar usando StandardScaler (média=0, desvio padrão=1)
        scaler = StandardScaler()
        df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])
        
        print(f"Features numéricas normalizadas: {cols_to_normalize}")
    else:
        print("Nenhuma feature numérica para normalizar.")
        df_normalized = df.copy()
    
    return df_normalized


def encode_categorical_features(df, categorical_cols):
    """Codificar features categóricas para uso em modelos de ML."""
    print("\nCodificando features categóricas...")
    
    # Remover colunas que não queremos codificar
    exclude_cols = ['Dis No', 'Comments', 'Appeal', 'Declaration', 'Aid Contribution']
    cols_to_encode = [col for col in categorical_cols if col not in exclude_cols]
    
    if cols_to_encode:
        # Criar um dataframe de cópia
        df_encoded = df.copy()
        
        # Para cada coluna categórica, aplicar one-hot encoding
        for col in cols_to_encode:
            # Selecionar apenas as categorias mais frequentes para colunas com muitas categorias
            if df_encoded[col].nunique() > 10:
                top_categories = df_encoded[col].value_counts().nlargest(10).index
                df_encoded[col] = df_encoded[col].apply(lambda x: x if x in top_categories else 'Other')
            
            # Criar variáveis dummy
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            
            # Adicionar ao dataframe
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            
            # Remover coluna original
            df_encoded.drop(col, axis=1, inplace=True)
        
        print(f"Features categóricas codificadas: {cols_to_encode}")
        print(f"Dimensões após codificação: {df_encoded.shape[0]} linhas e {df_encoded.shape[1]} colunas.")
    else:
        print("Nenhuma feature categórica para codificar.")
        df_encoded = df.copy()
    
    return df_encoded


def prepare_data_for_modeling(df):
    """Preparar os dados para modelagem, definindo features e alvo."""
    print("\nPreparando dados para modelagem...")
    
    # 1. Definir possíveis alvos para diferentes tipos de modelos
    potential_targets = {
        'mortality': 'Total Deaths',
        'affected': 'Total Affected',
        'damage': "Total Damage, Adjusted ('000 US$)",
        'combined': 'Combined_Impact',
        'binary_high_impact': 'High_Death_Count'
    }
    
    # 2. Verificar quais alvos estão disponíveis no dataframe
    available_targets = {}
    for target_name, target_col in potential_targets.items():
        if target_col in df.columns:
            available_targets[target_name] = target_col
            print(f"Alvo disponível: {target_name} ({target_col})")
    
    # 3. Identificar features disponíveis (todas as colunas exceto os alvos)
    feature_cols = [col for col in df.columns if col not in list(potential_targets.values())]
    
    # 4. Salvar dataframe processado
    processed_file = f"{PROCESSED_DIR}/processed_disasters_data.csv"
    df.to_csv(processed_file, index=False)
    print(f"Dataset processado salvo em: {processed_file}")
    
    # 5. Salvar arquivo de metadata para uso futuro
    metadata = {
        'feature_columns': feature_cols,
        'target_columns': available_targets,
        'dataset_shape': df.shape,
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'disaster_types': PRIORITY_DISASTERS
    }
    
    # Salvar metadata como CSV
    metadata_df = pd.DataFrame({
        'Key': list(metadata.keys()),
        'Value': [str(v) for v in metadata.values()]
    })
    metadata_file = f"{PROCESSED_DIR}/metadata.csv"
    metadata_df.to_csv(metadata_file, index=False)
    print(f"Metadata salvo em: {metadata_file}")
    
    return df, feature_cols, available_targets


def main():
    """Função principal para executar o fluxo de pré-processamento."""
    print("Iniciando pré-processamento do dataset de desastres naturais...")
    
    # 1. Carregar dados
    df = load_data()
    
    # 2. Filtrar para tipos de desastre prioritários
    df_filtered = filter_priority_disasters(df)
    
    # 3. Analisar valores ausentes
    missing_df = analyze_missing_values(df_filtered)
    
    # 4. Criar features temporais
    df_time = create_date_features(df_filtered)
    
    # 5. Tratar valores ausentes
    df_clean, categorical_cols, numerical_cols = handle_missing_values(df_time)
    
    # 6. Engenharia de features
    df_engineered = engineer_features(df_clean)
    
    # 7. Normalizar features numéricas
    df_normalized = normalize_numerical_features(df_engineered, numerical_cols)
    
    # 8. Codificar features categóricas
    df_encoded = encode_categorical_features(df_normalized, categorical_cols)
    
    # 9. Preparar dados para modelagem
    df_final, feature_cols, available_targets = prepare_data_for_modeling(df_encoded)
    
    print("\nPré-processamento concluído com sucesso!")
    print(f"Dataset final: {df_final.shape[0]} linhas e {df_final.shape[1]} colunas.")
    print(f"Features disponíveis: {len(feature_cols)}")
    print(f"Alvos disponíveis: {list(available_targets.keys())}")
    print(f"Resultados salvos em: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
