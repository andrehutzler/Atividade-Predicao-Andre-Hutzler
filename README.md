# Previsão de Série Temporal — Temperaturas mínimas diárias (Melbourne)

Este repositório entrega a atividade de predição de séries temporais comparando:
- Modelo clássico (sktime): ThetaForecaster dentro de um pipeline com **deseasonalização aditiva**.
- Modelo neural (deep learning): **LSTM** (Keras/TensorFlow) para previsão univariada.

A comparação é feita por **MAE** e **MASE**.

## Dataset
Kaggle: Daily minimum temperatures — Melbourne (1981–1990)  
https://www.kaggle.com/datasets/suprematism/daily-minimum-temperatures

O notebook primeiro tenta ler um arquivo local em `data/1_Daily_minimum_temps.csv`
(ou alternativamente `data/daily-minimum-temperatures.csv`).  
Se esse arquivo não existir, faz fallback automático para a URL pública com o mesmo CSV:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv


## Modelos implementados
**sktime (clássico):**
- Pipeline: Deseasonalizer(model="additive", sp=365) → ThetaForecaster(sp=365, deseasonalize=False).
- Motivo: modo aditivo é apropriado quando existem valores 0/negativos (evita erro de sazonalidade multiplicativa).

**LSTM (Keras/TensorFlow):**
- Normalização MinMax somente com dados de treino.
- WINDOW=30, HORIZON=1, EarlyStopping(patience=5).
- Previsão 1-step rolling no conjunto de teste.

## Métricas e justificativa
- **MAE (Mean Absolute Error):** intuitivo e na mesma unidade da série (°C); mede o erro médio absoluto.
- **MASE (Mean Absolute Scaled Error):** recomendado por Hyndman & Koehler (2006);
  escala o MAE do modelo pelo MAE de um forecast ingênuo (lag=1) calculado no treino.
  Comparável entre séries/horizontes e não sofre com problemas do MAPE perto de zero.

Interpretação rápida:
- MAE menor = melhor.
- MASE < 1: melhor que o ingênuo; ≈ 1: semelhante; > 1: pior que o ingênuo.

Referências:
- scikit-learn — `sklearn.metrics.mean_absolute_error`
- Hyndman, R. J., & Koehler, A. B. (2006). Another Look at Measures of Forecast Accuracy, IJF.

## Reprodutibilidade
Semente global: 42 (NumPy / TensorFlow). Pequenas variações podem ocorrer por operações não determinísticas (GPU/BLAS).

Autor: André Hutzler
