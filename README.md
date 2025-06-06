# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="https://raw.githubusercontent.com/lfusca/templateFiap/main/assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

## 👨‍🎓 Integrantes do Grupo

- RM559800 - [Jonas Felipe dos Santos Lima](https://www.linkedin.com/in/jonas-felipe-dos-santos-lima-b2346811b/)
- RM560173 - [Gabriel Ribeiro](https://www.linkedin.com/in/ribeirogab/)
- RM559926 - [Marcos Trazzini](https://www.linkedin.com/in/mstrazzini/)
- RM559645 - [Edimilson Ribeiro](https://www.linkedin.com/in/edimilson-ribeiro/)

## 👩‍🏫 Professores

### Coordenador(a)

- [André Godoi](https://www.linkedin.com/in/profandregodoi/)

---

## 📌 Entregas do Projeto

Este projeto representa a Global Solution FIAP 2025.1, um dashboard de monitoramento de desastres naturais que utiliza dados de sensores IoT, modelos de machine learning e redes neurais para prever, monitorar e mitigar os impactos de eventos extremos como inundações, tempestades e terremotos.

---

## 🛠 **Dashboard de Monitoramento de Desastres**

### 🎯 Objetivos

- Criar uma plataforma de monitoramento para visualizar dados de sensores em áreas de risco
- Analisar tendências históricas de desastres e realizar predições com modelos de aprendizado de máquina
- Gerenciar dispositivos IoT de monitoramento remotamente
- Emitir e acompanhar alertas de risco com base em dados em tempo real
- Visualizar geograficamente zonas de risco e dispositivos de monitoramento

---

### 📁 Estrutura de Pastas/Arquivos

```plaintext
/global-solution-2
├── docs/                          # Documentação do projeto
│   ├── PROJECT_BRIEF.md          # Descrição do tema e requisitos da Global Solution
│   ├── project-overview.md        # Visão geral do projeto
│   ├── scope_and_requirements.md  # Escopo e requisitos detalhados
│   └── tasks/                    # Documentação de tarefas específicas
│
├── src/                           # Código-fonte do projeto
│   ├── dashboard/                 # Aplicação Streamlit para o dashboard
│   │   ├── app.py                # Ponto de entrada principal do dashboard
│   │   ├── dashboard.py           # Implementação da página inicial do dashboard
│   │   ├── pages/                # Páginas do dashboard multi-página
│   │   │   ├── __init__.py       # Define o diretório como pacote Python
│   │   │   ├── alerts.py         # Página de gerenciamento de alertas
│   │   │   ├── analytics.py      # Página de análises e predições
│   │   │   ├── devices.py        # Página de gerenciamento de dispositivos IoT
│   │   │   └── mapview.py        # Visualização geográfica de zonas de risco
│   │   └── utils/                # Utilitários para o dashboard
│   │       ├── __init__.py       # Define o diretório como pacote Python
│   │       ├── data_loader.py    # Carregamento e processamento de dados
│   │       └── model_loader.py   # Carregamento e utilização de modelos de ML
│   ├── esp32/                    # Código para dispositivos ESP32
│   │   ├── circuit_diagram.txt   # Diagrama de circuito para o hardware
│   │   ├── disaster_monitoring_system.ino  # Código Arduino para o ESP32
│   │   └── simulator.py         # Simulador de dados de sensores
│   ├── data_preprocessing.py    # Preparação e limpeza de dados históricos
│   ├── exploratory_analysis.py  # Análise exploratória de dados de desastres
│   ├── model_development.py     # Desenvolvimento de modelos preditivos
│   ├── model_example.py         # Implementação de exemplo de modelo de ML
│   ├── neural_network_development.py    # Desenvolvimento da rede neural
│   ├── neural_network_functions.py     # Funções utilizadas pela rede neural
│   ├── neural_network_main.py          # Script principal da rede neural
│   └── neural_network_quick_test.py    # Testes rápidos da rede neural
└── README.md                    # Este arquivo
```

---

### 🔧 Como Executar

#### Configuração Inicial

1. Clone este repositório:

   ```bash
   git clone https://github.com/seu-usuario/global-solution-2.git
   cd global-solution-2
   ```

2. Instale as dependências Python:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure as variáveis de ambiente (se necessário):

   ```bash
   # Crie um arquivo .env na pasta raíz do projeto
   cp .env.example .env
   # Edite o arquivo .env com suas credenciais
   ```

#### Executando o Dashboard

1. Inicie o dashboard principal:

   ```bash
   cd src/dashboard
   streamlit run app.py
   ```

2. Navegue pelo menu lateral para acessar as diferentes páginas do dashboard.

### 💻 Tecnologias Utilizadas

- **Linguagens de Programação:**
  - Python 3.x (principal linguagem do projeto)
  - SQL (para consultas de dados em banco de dados)

- **Bibliotecas e Frameworks:**
  - **Python**:
    - Streamlit (para dashboard interativo)
    - Pandas, NumPy (para manipulação de dados)
    - Plotly Express (para visualizações interativas)
    - PyDeck (para visualizações geográficas)
    - Scikit-learn (para Machine Learning)
    - JSON (para formatação de dados)

- **Ferramentas e Serviços:**
  - **Hardware**:
    - Sensores IoT (umidade, temperatura, etc.)
    - Dispositivos de telemetria
  - **Outros**:
    - Git (controle de versão)
    - GitHub (hospedagem do repositório)

---

## 📋 Funcionalidades do Dashboard

### 📊 Página Principal

A página principal do dashboard apresenta uma visão geral do sistema, com:

- Indicadores chave de performance (KPIs)
- Resumo de dispositivos ativos e inativos
- Alertas recentes e não resolvidos
- Estatísticas gerais sobre áreas monitoradas

### 🖥️ Dispositivos

A página de Dispositivos permite:

- Visualizar todos os dispositivos de monitoramento cadastrados
- Filtrar dispositivos por status (online/offline/warning)
- Ver detalhes de cada dispositivo, incluindo:
  - Data da última leitura
  - Nível de bateria
  - Dados dos sensores
- Executar ações de gerenciamento remoto:
  - Reiniciar dispositivo
  - Calibrar sensores
  - Atualizar firmware

### 📈 Analytics

A página de Analytics oferece:

- Análise temporal de desastres históricos
- Impacto por tipo de desastre
- Predições baseadas em modelos de Machine Learning
- Visualização da importância de diferentes variáveis na previsão de desastres
- Resumos estatísticos sobre dados coletados

### 🗺️ Mapa

A página de visualização geográfica permite:

- Ver a distribuição de dispositivos de monitoramento em um mapa interativo
- Filtrar dispositivos por status
- Visualizar zonas de risco classificadas por nível de perigo
- Acessar detalhes sobre cada zona de risco
- Reportar novas áreas de risco

### ⚠️ Alertas

O sistema de alertas possibilita:

- Visualizar todos os alertas ativos
- Filtrar alertas por nível de severidade
- Gerenciar o status de alertas (em análise, resolvido, etc.)
- Atribuir responsáveis para resolver cada alerta
- Visualizar histórico de alertas passados

---

## 🔄 Integrações Futuras

- **APIs Meteorológicas**: Integração com serviços de previsão do tempo para alertas antecipados
- **Sistema de Notificações**: Envio de alertas via e-mail ou SMS para equipes de resposta
- **Banco de Dados em Nuvem**: Migração para armazenamento na nuvem para melhor escalabilidade
- **Inteligência Artificial**: Aprimoramento dos modelos preditivos com técnicas avançadas de ML

## 🎥 Demonstração

[Link para vídeo demonstrativo do projeto] (a ser adicionado)

## 📋 Licença

Este projeto segue o modelo de licença da FIAP e está licenciado sob **Attribution 4.0 International**. Para mais informações, consulte o [MODELO GIT FIAP](https://github.com/agodoi/template).
