"""Helper functions for data analysis and visualization."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_output_dir(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    return dir_path


def save_plot(fig, filename, output_dir="results/plots", dpi=300, bbox_inches="tight"):
    """Save matplotlib figure to specified directory."""
    create_output_dir(output_dir)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Plot saved to {filepath}")
    return filepath


def get_missing_info(df):
    """Get information about missing values in DataFrame."""
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percent Missing': missing_percent
    })
    return missing_info[missing_info['Missing Values'] > 0].sort_values('Percent Missing', ascending=False)


def plot_correlation_matrix(df, columns=None, figsize=(12, 10), cmap='coolwarm'):
    """Plot correlation matrix for selected numeric columns."""
    if columns is None:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[columns].select_dtypes(include=[np.number])
    
    corr = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5)
    plt.title('Correlation Matrix of Numeric Features', fontsize=15)
    
    return plt.gcf()


def plot_categorical_distribution(df, column, top_n=10, figsize=(12, 6)):
    """Plot distribution of categorical variable."""
    plt.figure(figsize=figsize)
    
    # Get value counts and take top N
    value_counts = df[column].value_counts().head(top_n)
    
    # Create barplot
    ax = sns.barplot(x=value_counts.index, y=value_counts.values)
    
    # Add labels and title
    plt.title(f'Top {top_n} {column} Distribution', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add count labels on top of bars
    for i, count in enumerate(value_counts.values):
        ax.text(i, count + (max(value_counts.values) * 0.01), f'{count:,}', 
                ha='center', va='bottom', fontsize=10)
    
    return plt.gcf()


def plot_numeric_distribution(df, column, bins=30, figsize=(12, 6)):
    """Plot distribution of numeric variable."""
    plt.figure(figsize=figsize)
    
    # Create distribution plot
    sns.histplot(df[column].dropna(), bins=bins, kde=True)
    
    # Add summary statistics
    mean_val = df[column].mean()
    median_val = df[column].median()
    
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
    
    # Add labels and title
    plt.title(f'Distribution of {column}', fontsize=15)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()


def plot_time_series(df, date_column, value_column, freq='Y', figsize=(14, 7)):
    """Plot time series data."""
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by date and aggregate
    time_series = df.groupby(pd.Grouper(key=date_column, freq=freq))[value_column].sum().reset_index()
    
    plt.figure(figsize=figsize)
    plt.plot(time_series[date_column], time_series[value_column], marker='o')
    
    # Add labels and title
    plt.title(f'{value_column} over Time ({freq} frequency)', fontsize=15)
    plt.xlabel(date_column)
    plt.ylabel(value_column)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()
