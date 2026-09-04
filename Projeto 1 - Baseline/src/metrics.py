"""
metrics.py
==========
Implementa as 4 métricas de avaliação exigidas para a tarefa de
regressão: MAE, MSE, RMSE e R².

Aceitam tanto arrays numpy quanto tensores PyTorch como entrada (tensores
são convertidos automaticamente para numpy).
"""

import numpy as np
import torch


def _to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def mae(y_true, y_pred):
    """Erro Absoluto Médio: média de |y_true - y_pred|.
    Interpretação: em média, o quanto a predição erra, na MESMA unidade
    de y. É robusto a outliers (não eleva o erro ao quadrado)."""
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    """Erro Quadrático Médio: média de (y_true - y_pred)^2.
    Penaliza mais fortemente erros grandes (por causa do quadrado).
    É também a função de perda usada para treinar a rede (sem os termos
    extras de regularização L1/L2, que só entram durante o treino)."""
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    """Raiz do MSE. Mesma unidade de y (ao contrário do MSE), o que
    facilita a interpretação prática do erro típico do modelo."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true, y_pred):
    """
    Coeficiente de Determinação R².
    R² = 1 - (soma dos quadrados dos resíduos) / (soma dos quadrados
              totais em torno da média de y_true)

    Interpretação: fração da variância de y que o modelo consegue
    explicar. R²=1 -> predição perfeita. R²=0 -> modelo não é melhor do
    que simplesmente prever a média de y. R²<0 -> modelo é PIOR do que
    prever a média (pode acontecer quando o modelo generaliza mal).
    """
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def all_metrics(y_true, y_pred):
    """Retorna um dicionário com as 4 métricas de uma vez."""
    return {
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
