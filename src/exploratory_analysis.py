#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis for Natural Disasters Dataset

This script performs a comprehensive exploratory analysis of the natural disasters dataset,
including data cleaning, visualization, and insights extraction to help identify
the most relevant disaster type for the project focus.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import helper functions
# Usamos import relativo para evitar problemas de módulo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import (
    get_missing_info, save_plot, create_output_dir,
    plot_correlation_matrix, plot_categorical_distribution, 
    plot_numeric_distribution, plot_time_series
)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Define paths
DATASET_PATH = '/Users/gabrielribeiro/www/fiap/global-solution-2/dataset.xlsx'
REPORTS_DIR = 'results/reports'
PLOTS_DIR = 'results/plots'

# Create directories if they don't exist
create_output_dir(REPORTS_DIR)
create_output_dir(PLOTS_DIR)


def load_data(file_path=DATASET_PATH):
    """Load the dataset from Excel file."""
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def examine_data_structure(df):
    """Examine the structure of the dataset."""
    print("\n===== Data Structure =====")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nColumns and data types:")
    print(df.dtypes)
    
    print("\nSummary statistics for numeric columns:")
    print(df.describe())
    
    # Check for missing values
    missing_info = get_missing_info(df)
    print("\nMissing values:")
    print(missing_info)
    
    # Save missing values info to CSV
    missing_info.to_csv(f"{REPORTS_DIR}/missing_values_summary.csv")
    
    # Create a plot of missing values
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=missing_info.index, y=missing_info['Percent Missing'])
    plt.title('Percentage of Missing Values by Column', fontsize=15)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylabel('Percent Missing')
    save_plot(plt.gcf(), "missing_values_by_column.png", PLOTS_DIR)
    plt.close()
    
    return missing_info


def analyze_disaster_types(df):
    """Analyze the distribution of disaster types and their impacts."""
    print("\n===== Disaster Types Analysis =====")
    
    # Disaster Group distribution
    print("\nDisaster Group distribution:")
    disaster_group_counts = df['Disaster Group'].value_counts()
    print(disaster_group_counts)
    
    # Plot Disaster Group distribution
    fig = plot_categorical_distribution(df, 'Disaster Group')
    save_plot(fig, "disaster_group_distribution.png", PLOTS_DIR)
    plt.close()
    
    # Disaster Type distribution
    print("\nDisaster Type distribution:")
    disaster_type_counts = df['Disaster Type'].value_counts().head(15)
    print(disaster_type_counts)
    
    # Plot Disaster Type distribution
    fig = plot_categorical_distribution(df, 'Disaster Type', top_n=15)
    save_plot(fig, "disaster_type_distribution.png", PLOTS_DIR)
    plt.close()
    
    # Analyze total deaths by disaster type
    disaster_deaths = df.groupby('Disaster Type')['Total Deaths'].sum().sort_values(ascending=False)
    print("\nTotal deaths by disaster type:")
    print(disaster_deaths.head(10))
    
    # Plot total deaths by disaster type
    plt.figure(figsize=(12, 6))
    disaster_deaths.head(10).plot(kind='bar')
    plt.title('Total Deaths by Disaster Type (Top 10)', fontsize=15)
    plt.tight_layout()
    plt.ylabel('Total Deaths')
    save_plot(plt.gcf(), "total_deaths_by_disaster_type.png", PLOTS_DIR)
    plt.close()
    
    # Analyze total affected by disaster type
    disaster_affected = df.groupby('Disaster Type')['Total Affected'].sum().sort_values(ascending=False)
    print("\nTotal affected by disaster type:")
    print(disaster_affected.head(10))
    
    # Plot total affected by disaster type
    plt.figure(figsize=(12, 6))
    disaster_affected.head(10).plot(kind='bar')
    plt.title('Total Affected by Disaster Type (Top 10)', fontsize=15)
    plt.tight_layout()
    plt.ylabel('Total Affected')
    save_plot(plt.gcf(), "total_affected_by_disaster_type.png", PLOTS_DIR)
    plt.close()
    
    # Analyze total damage by disaster type
    disaster_damage = df.groupby('Disaster Type')["Total Damage, Adjusted ('000 US$)"].sum().sort_values(ascending=False)
    print("\nTotal damage by disaster type (in '000 US$):")
    print(disaster_damage.head(10))
    
    # Plot total damage by disaster type
    plt.figure(figsize=(12, 6))
    disaster_damage.head(10).plot(kind='bar')
    plt.title("Total Damage by Disaster Type (Top 10) in '000 US$", fontsize=15)
    plt.tight_layout()
    plt.ylabel("Total Damage ('000 US$)")
    save_plot(plt.gcf(), "total_damage_by_disaster_type.png", PLOTS_DIR)
    plt.close()
    
    return disaster_group_counts, disaster_type_counts, disaster_deaths, disaster_affected, disaster_damage


def analyze_geographic_distribution(df):
    """Analyze the geographic distribution of disasters."""
    print("\n===== Geographic Distribution Analysis =====")
    
    # Region distribution
    print("\nRegion distribution:")
    region_counts = df['Region'].value_counts()
    print(region_counts)
    
    # Plot Region distribution
    fig = plot_categorical_distribution(df, 'Region')
    save_plot(fig, "region_distribution.png", PLOTS_DIR)
    plt.close()
    
    # Country distribution (top 20)
    print("\nTop 20 countries by number of disasters:")
    country_counts = df['Country'].value_counts().head(20)
    print(country_counts)
    
    # Plot Country distribution
    fig = plot_categorical_distribution(df, 'Country', top_n=20)
    save_plot(fig, "country_distribution.png", PLOTS_DIR)
    plt.close()
    
    # Analyze disaster types by region
    disaster_by_region = pd.crosstab(df['Region'], df['Disaster Type'])
    print("\nDisaster types by region (top 5 disaster types):")
    top_disasters = df['Disaster Type'].value_counts().head(5).index
    print(disaster_by_region[top_disasters])
    
    # Plot disaster types by region (heatmap)
    plt.figure(figsize=(14, 8))
    sns.heatmap(disaster_by_region[top_disasters], cmap='YlOrRd', annot=True, fmt='d')
    plt.title('Distribution of Top 5 Disaster Types by Region', fontsize=15)
    plt.tight_layout()
    save_plot(plt.gcf(), "disaster_types_by_region_heatmap.png", PLOTS_DIR)
    plt.close()
    
    return region_counts, country_counts, disaster_by_region


def analyze_temporal_trends(df):
    """Analyze temporal trends in disaster occurrences and impacts."""
    print("\n===== Temporal Trends Analysis =====")
    
    # Create a start date column combining year, month, day
    df_temp = df.copy()
    # Fill missing month and day with defaults (January 1st)
    df_temp['Start Month'] = df_temp['Start Month'].fillna(1)
    df_temp['Start Day'] = df_temp['Start Day'].fillna(1)
    
    # Create datetime column
    df_temp['Start Date'] = pd.to_datetime(
        df_temp[['Start Year', 'Start Month', 'Start Day']].astype(int).astype(str).agg('-'.join, axis=1),
        errors='coerce'
    )
    
    # Group disasters by year
    yearly_disasters = df_temp.groupby(df_temp['Start Year'])['DisNo.'].count()
    print("\nNumber of disasters by year:")
    print(yearly_disasters.tail(10))  # Show last 10 years
    
    # Plot yearly trend
    plt.figure(figsize=(14, 6))
    yearly_disasters.plot()
    plt.title('Number of Disasters per Year', fontsize=15)
    plt.xlabel('Year')
    plt.ylabel('Number of Disasters')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(plt.gcf(), "disasters_per_year.png", PLOTS_DIR)
    plt.close()
    
    # Group disasters by year and disaster type
    disaster_types_yearly = df_temp.pivot_table(
        index='Start Year', 
        columns='Disaster Type', 
        values='DisNo.', 
        aggfunc='count'
    ).fillna(0)
    
    # Select top 5 disaster types
    top_5_types = df['Disaster Type'].value_counts().head(5).index
    disaster_types_yearly_top5 = disaster_types_yearly[top_5_types]
    
    print("\nYearly trend of top 5 disaster types:")
    print(disaster_types_yearly_top5.tail(10))  # Show last 10 years
    
    # Plot yearly trend by disaster type
    plt.figure(figsize=(14, 8))
    disaster_types_yearly_top5.plot()
    plt.title('Yearly Trend of Top 5 Disaster Types', fontsize=15)
    plt.xlabel('Year')
    plt.ylabel('Number of Disasters')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Disaster Type')
    plt.tight_layout()
    save_plot(plt.gcf(), "disaster_types_yearly_trend.png", PLOTS_DIR)
    plt.close()
    
    # Analyze total deaths over time
    yearly_deaths = df_temp.groupby(df_temp['Start Year'])['Total Deaths'].sum()
    
    # Plot yearly deaths
    plt.figure(figsize=(14, 6))
    yearly_deaths.plot()
    plt.title('Total Deaths from Disasters per Year', fontsize=15)
    plt.xlabel('Year')
    plt.ylabel('Total Deaths')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(plt.gcf(), "total_deaths_per_year.png", PLOTS_DIR)
    plt.close()
    
    return yearly_disasters, disaster_types_yearly, yearly_deaths


def analyze_impact_metrics(df):
    """Analyze the impact metrics (deaths, injuries, damages, etc.)."""
    print("\n===== Impact Metrics Analysis =====")
    
    # Select impact columns
    impact_columns = ['Total Deaths', 'No. Injured', 'No. Affected', 
                     'No. Homeless', 'Total Affected', 
                     'Total Damage (\'000 US$)', 'Total Damage, Adjusted (\'000 US$)']
    
    # Summary statistics for impact metrics
    impact_stats = df[impact_columns].describe()
    print("\nSummary statistics for impact metrics:")
    print(impact_stats)
    
    # Save impact statistics to CSV
    impact_stats.to_csv(f"{REPORTS_DIR}/impact_metrics_summary.csv")
    
    # Distribution of total deaths
    fig = plot_numeric_distribution(df, 'Total Deaths')
    save_plot(fig, "total_deaths_distribution.png", PLOTS_DIR)
    plt.close()
    
    # Distribution of total affected
    fig = plot_numeric_distribution(df, 'Total Affected')
    save_plot(fig, "total_affected_distribution.png", PLOTS_DIR)
    plt.close()
    
    # Distribution of total damage
    fig = plot_numeric_distribution(df, 'Total Damage, Adjusted (\'000 US$)')
    save_plot(fig, "total_damage_distribution.png", PLOTS_DIR)
    plt.close()
    
    # Correlation between impact metrics
    correlation_fig = plot_correlation_matrix(df, impact_columns)
    save_plot(correlation_fig, "impact_metrics_correlation.png", PLOTS_DIR)
    plt.close()
    
    return impact_stats


def identify_disaster_type_for_focus(df, disaster_deaths, disaster_affected, disaster_damage):
    """Identify and recommend a disaster type to focus on for the project."""
    print("\n===== Disaster Type Selection for Project Focus =====")
    
    # Create a scoring system for each disaster type
    disaster_types = df['Disaster Type'].unique()
    
    # Initialize a DataFrame to hold the scores
    scores_df = pd.DataFrame(index=disaster_types)
    
    # Add rankings for different metrics (lower rank = higher impact)
    scores_df['Death Rank'] = pd.Series(disaster_deaths).rank(ascending=False)
    scores_df['Affected Rank'] = pd.Series(disaster_affected).rank(ascending=False)
    scores_df['Damage Rank'] = pd.Series(disaster_damage).rank(ascending=False)
    
    # Fill NaN with worst rank + 1
    max_rank = len(disaster_types) + 1
    scores_df = scores_df.fillna(max_rank)
    
    # Calculate an overall score (weighted average of ranks)
    # Weights: Deaths (40%), Affected (30%), Damage (30%)
    scores_df['Overall Score'] = (
        0.4 * scores_df['Death Rank'] + 
        0.3 * scores_df['Affected Rank'] + 
        0.3 * scores_df['Damage Rank']
    )
    
    # Sort by overall score (lower is better)
    scores_df = scores_df.sort_values('Overall Score')
    
    # Count occurrences of each disaster type
    disaster_frequency = df['Disaster Type'].value_counts()
    scores_df['Frequency'] = disaster_frequency
    scores_df['Frequency'] = scores_df['Frequency'].fillna(0)
    
    print("\nTop 10 disaster types by impact score:")
    print(scores_df.head(10))
    
    # Save rankings to CSV
    scores_df.to_csv(f"{REPORTS_DIR}/disaster_type_rankings.csv")
    
    # Plot top 10 disaster types by score
    plt.figure(figsize=(12, 6))
    scores_df.head(10)['Overall Score'].plot(kind='bar')
    plt.title('Top 10 Disaster Types by Impact Score (Lower is Better)', fontsize=15)
    plt.tight_layout()
    plt.ylabel('Impact Score')
    save_plot(plt.gcf(), "disaster_types_by_impact_score.png", PLOTS_DIR)
    plt.close()
    
    # Recommended disaster type (top 3)
    print("\nRecommended disaster types to focus on (top 3):")
    for i, (disaster_type, row) in enumerate(scores_df.head(3).iterrows()):
        print(f"{i+1}. {disaster_type}")
        print(f"   Deaths Rank: {row['Death Rank']:.0f}, Affected Rank: {row['Affected Rank']:.0f}, Damage Rank: {row['Damage Rank']:.0f}")
        print(f"   Frequency: {row['Frequency']:.0f} occurrences")
        print(f"   Overall Impact Score: {row['Overall Score']:.2f} (lower is better)")
    
    return scores_df


def generate_analysis_report(df, scores_df):
    """Generate a comprehensive analysis report in Markdown format."""
    # Get top disaster type
    top_disaster = scores_df.index[0]
    
    # Create report content
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Análise Exploratória do Dataset de Desastres Naturais

*Relatório gerado em: {now}*

## 1. Visão Geral do Dataset

O dataset analisado contém informações sobre desastres naturais ocorridos em todo o mundo, com {df.shape[0]} registros e {df.shape[1]} colunas. As informações incluem tipos de desastres, localizações geográficas, datas, métricas de impacto (mortes, feridos, afetados, danos) entre outros dados.

### 1.1 Principais Estatísticas

- **Número total de desastres registrados**: {df.shape[0]}
- **Número de tipos de desastres diferentes**: {df['Disaster Type'].nunique()}
- **Número de países afetados**: {df['Country'].nunique()}
- **Período coberto**: {df['Start Year'].min()} a {df['Start Year'].max()}

## 2. Distribuição de Tipos de Desastres

### 2.1 Grupos de Desastres

Os desastres são classificados em {df['Disaster Group'].nunique()} grupos principais:

{df['Disaster Group'].value_counts().to_markdown()}

### 2.2 Tipos de Desastres Mais Comuns

Os 10 tipos de desastres mais comuns no dataset são:

{df['Disaster Type'].value_counts().head(10).to_markdown()}

## 3. Análise de Impacto

### 3.1 Mortes por Tipo de Desastre

Os tipos de desastres que causaram mais mortes:

{df.groupby('Disaster Type')['Total Deaths'].sum().sort_values(ascending=False).head(10).to_markdown()}

### 3.2 Pessoas Afetadas por Tipo de Desastre

Os tipos de desastres que afetaram mais pessoas:

{df.groupby('Disaster Type')['Total Affected'].sum().sort_values(ascending=False).head(10).to_markdown()}

### 3.3 Danos Materiais por Tipo de Desastre

Os tipos de desastres que causaram mais danos materiais (em milhares de US$):

{df.groupby('Disaster Type')["Total Damage, Adjusted ('000 US$)"].sum().sort_values(ascending=False).head(10).to_markdown()}

## 4. Distribuição Geográfica

### 4.1 Regiões Mais Afetadas

{df['Region'].value_counts().to_markdown()}

### 4.2 Países Mais Afetados (Top 10)

{df['Country'].value_counts().head(10).to_markdown()}

## 5. Tendências Temporais

A análise da distribuição temporal dos desastres revela padrões importantes sobre a frequência e evolução dos eventos ao longo do tempo. Os gráficos gerados mostram as tendências de ocorrência de desastres por ano.

## 6. Tipo de Desastre Recomendado para o Projeto

Com base na análise integrada de múltiplos fatores (mortalidade, número de afetados, danos materiais e frequência), o tipo de desastre recomendado para foco do projeto é:

**{top_disaster}**

Justificativa para esta escolha:

1. **Impacto em Vidas**: Este tipo de desastre apresenta um alto ranking em termos de mortalidade.
2. **Impacto em População Afetada**: Afeta um número significativo de pessoas além das fatalidades diretas.
3. **Impacto Econômico**: Causa danos materiais consideráveis.
4. **Relevância Estatística**: Possui ocorrências suficientes no dataset para permitir uma análise robusta e treinamento de modelos.

## 7. Próximos Passos

Com base nesta análise exploratória, recomenda-se:

1. Filtrar o dataset para focar nos dados relacionados a {top_disaster}
2. Realizar análise mais aprofundada das características específicas deste tipo de desastre
3. Identificar variáveis preditivas que possam ser utilizadas em modelos de machine learning
4. Começar o desenvolvimento de um modelo para previsão, monitoramento ou mitigação deste tipo de desastre
5. Investigar fontes de dados adicionais específicas para este tipo de desastre

## 8. Anexos

Os seguintes arquivos de visualização foram gerados e estão disponíveis na pasta 'results/plots':

- Distribuição de tipos de desastres
- Impacto por tipo de desastre
- Distribuição geográfica
- Tendências temporais
- Correlações entre métricas de impacto
"""
    
    # Save report to markdown file
    report_path = f"{REPORTS_DIR}/exploratory_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nComprehensive analysis report saved to {report_path}")
    return report_path


def main():
    """Main function to execute the exploratory analysis."""
    print("Starting exploratory analysis of natural disasters dataset...\n")
    
    # Load data
    df = load_data()
    
    # Examine data structure
    missing_info = examine_data_structure(df)
    
    # Analyze disaster types
    disaster_group_counts, disaster_type_counts, disaster_deaths, disaster_affected, disaster_damage = analyze_disaster_types(df)
    
    # Analyze geographic distribution
    region_counts, country_counts, disaster_by_region = analyze_geographic_distribution(df)
    
    # Analyze temporal trends
    yearly_disasters, disaster_types_yearly, yearly_deaths = analyze_temporal_trends(df)
    
    # Analyze impact metrics
    impact_stats = analyze_impact_metrics(df)
    
    # Identify disaster type for focus
    scores_df = identify_disaster_type_for_focus(df, disaster_deaths, disaster_affected, disaster_damage)
    
    # Generate analysis report
    report_path = generate_analysis_report(df, scores_df)
    
    print("\nExploratory analysis completed successfully!")
    print(f"Results saved to {REPORTS_DIR} and {PLOTS_DIR} directories.")
    print(f"Comprehensive report available at: {report_path}")


if __name__ == "__main__":
    main()
