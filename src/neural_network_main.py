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
