# Tarefa 6: Implementação com ESP32 e Sensores

## Objetivo
Desenvolve a parte de hardware do projeto utilizando ESP32 e sensores apropriados para monitoramento de condições ambientais relacionadas ao tipo de desastre natural escolhido, atendendo aos requisitos mínimos da Global Solution.

## Atividades
1. Selecionar os sensores adequados para o tipo de desastre escolhido (temperatura, umidade, pressão, movimento, etc.)
2. Projetar o circuito eletrônico integrando ESP32 e sensores selecionados
3. Montar o protótipo físico do sistema
4. Desenvolver o firmware para o ESP32 (em C/C++ ou MicroPython)
5. Implementar a coleta de dados dos sensores
6. Configurar a comunicação WiFi/BLE para transmissão de dados
7. Implementar sistema de alerta baseado em thresholds
8. Realizar testes de funcionamento e calibração
9. Otimizar o consumo de energia para uso em campo

## Entregáveis
- Esquema do circuito eletrônico (diagrama)
- Protótipo físico funcional
- Código-fonte do firmware para ESP32
- Documentação detalhada do hardware e firmware
- Relatório de testes e calibração
- Manual de uso e instalação

## Recursos Necessários
- ESP32 DevKit ou NodeMCU
- Sensores adequados ao tipo de desastre (DHT22, BMP280, acelerômetros, etc.)
- Componentes eletrônicos (resistores, capacitores, etc.)
- Protoboard e jumpers
- Arduino IDE ou PlatformIO
- Bibliotecas para os sensores utilizados

## Critérios de Aceitação
- Sistema capaz de coletar dados relevantes dos sensores
- Firmware 100% operacional conforme exigido na GS
- Comunicação funcional para transmissão de dados
- Sistema de alerta implementado e testado
- Baixo consumo de energia para operação em campo
