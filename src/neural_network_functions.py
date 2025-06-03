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
