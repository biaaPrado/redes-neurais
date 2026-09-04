"""
experiments.py
===============
Script principal do projeto. Treina:
  1) o baseline (SGD puro, sem L1/L2/dropout/momentum);
  2) baseline + L1;
  3) baseline + L2;
  4) baseline + Dropout;
  5) baseline + Momentum;

usando SEMPRE a mesma divisão de dados (fixada por data_utils.py), a
MESMA arquitetura e os MESMOS pesos iniciais (baseline_config.py) - a
ÚNICA diferença entre os 5 modelos é o componente adicional estudado.

Ao final, o script:
  - gera um gráfico de evolução do treino (loss de treino x validação
    por época) para CADA modelo;
  - gera um gráfico comparativo sobrepondo a curva de validação dos 5
    modelos;
  - gera um gráfico da função aprendida por cada modelo sobreposta aos
    dados reais (fácil de visualizar aqui pois x é 1-dimensional);
  - calcula MAE, MSE, RMSE e R² em treino, validação e teste para cada
    modelo, e salva uma tabela comparativa final.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import create_or_load_fixed_split, standardize_to_tensors, RANDOM_SEED
from model import MLP
from trainer import train
from metrics import all_metrics
from baseline_config import ARCHITECTURE, LEARNING_RATE, EPOCHS, BATCH_SIZE, INIT_SEED

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

SMOOTH_WINDOW = 51  # tamanho da janela da média móvel usada só para plotar


def moving_average(values, window=SMOOTH_WINDOW):
    """
    Suaviza uma curva ruidosa usando média móvel simples.

    Por que suavizar?
    -------------------
    Com apenas 30 amostras de treino, batch_size=8 e 3000 épocas, a perda
    de VALIDAÇÃO (medida sobre só 30 pontos) varia muito de época para
    época - não porque o treino esteja "quebrado", mas porque a cada
    época os pesos mudam um pouco (mini-batch SGD) e o MSE calculado
    sobre uma amostra pequena é naturalmente ruidoso. Isso dificulta
    enxergar a TENDÊNCIA real da curva num gráfico com milhares de
    pontos brutos.

    A suavização NÃO altera nenhum resultado numérico do projeto (as
    métricas finais em results/tables/ continuam calculadas sobre o
    melhor checkpoint, sem suavização nenhuma) - serve apenas para
    tornar os gráficos mais legíveis. Por isso, nos gráficos, sempre
    desenhamos a curva bruta (fina, translúcida) por trás da curva
    suavizada (grossa, opaca): quem quiser ver o ruído real, consegue.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # 'same' mantém o mesmo tamanho de vetor; nas bordas, o kernel "sai"
    # do vetor, então corrigimos dividindo pelo número real de vizinhos
    # usados em cada posição (em vez de sempre assumir `window` vizinhos).
    smoothed = np.convolve(values, kernel, mode="same")
    weight_used = np.convolve(np.ones_like(values), kernel, mode="same")
    return smoothed / weight_used


# ----------------------------------------------------------------------
# Valores de intensidade escolhidos em ablation_hparam_search.py
# ----------------------------------------------------------------------
L1_LAMBDA = 0.0001
L2_LAMBDA = 0.0001
DROPOUT_RATE = 0.1
MOMENTUM = 0.7

# ----------------------------------------------------------------------
# Definição dos 5 modelos do estudo. Repare que TODOS usam a mesma
# ARCHITECTURE, o mesmo LEARNING_RATE, os mesmos EPOCHS/BATCH_SIZE e a
# mesma INIT_SEED (pesos iniciais idênticos) - só o dicionário de
# hiperparâmetros extras muda.
# ----------------------------------------------------------------------
EXPERIMENTS = {
    "Baseline":          dict(dropout_rate=0.0, momentum=0.0, weight_decay=0.0, l1_lambda=0.0),
    "Baseline + L1":     dict(dropout_rate=0.0, momentum=0.0, weight_decay=0.0, l1_lambda=L1_LAMBDA),
    "Baseline + L2":     dict(dropout_rate=0.0, momentum=0.0, weight_decay=L2_LAMBDA, l1_lambda=0.0),
    "Baseline + Dropout": dict(dropout_rate=DROPOUT_RATE, momentum=0.0, weight_decay=0.0, l1_lambda=0.0),
    "Baseline + Momentum": dict(dropout_rate=0.0, momentum=MOMENTUM, weight_decay=0.0, l1_lambda=0.0),
}


def run_all_experiments():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_projeto1.csv")
    splits = create_or_load_fixed_split(raw_path)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = standardize_to_tensors(
        splits["train"], splits["val"], splits["test"])

    all_histories = {}
    all_models = {}
    metrics_rows = []

    for name, cfg in EXPERIMENTS.items():
        print(f"\n=== Treinando: {name} ===")
        model = MLP(ARCHITECTURE, dropout_rate=cfg["dropout_rate"], seed=INIT_SEED)
        history = train(model, X_train, y_train, X_val, y_val,
                         lr=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
                         l1_lambda=cfg["l1_lambda"], seed=RANDOM_SEED,
                         verbose=True, log_every=500, restore_best=True)

        print(f"  Melhor época (checkpoint): {history['best_epoch']} "
              f"| loss val nesse ponto: {history['best_val_loss']:.4f}")

        all_histories[name] = history
        all_models[name] = model

        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            y_pred_val = model(X_val)
            y_pred_test = model(X_test)

        for split_name, y_true, y_pred in [
            ("treino", y_train, y_pred_train),
            ("validação", y_val, y_pred_val),
            ("teste", y_test, y_pred_test),
        ]:
            m = all_metrics(y_true, y_pred)
            metrics_rows.append({
                "modelo": name, "conjunto": split_name,
                "MAE": m["MAE"], "MSE": m["MSE"], "RMSE": m["RMSE"], "R2": m["R2"],
                "melhor_epoca": history["best_epoch"],
            })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(TABLES_DIR, "metricas_comparativas.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[experiments] Tabela de métricas salva em: {metrics_path}")

    # ------------------------------------------------------------------
    # Gráficos: evolução do treino (treino x validação), um por modelo
    # ------------------------------------------------------------------
    for name, history in all_histories.items():
        epochs_range = np.arange(1, len(history["train_loss"]) + 1)
        train_raw = np.array(history["train_loss"])
        val_raw = np.array(history["val_loss"])
        train_smooth = moving_average(train_raw)
        val_smooth = moving_average(val_raw)

        plt.figure(figsize=(7, 4.5))
        # Curvas brutas: finas e translúcidas, só para mostrar que o
        # ruído existe (é esperado, dado o tamanho pequeno dos conjuntos).
        plt.plot(epochs_range, train_raw, color="tab:blue", alpha=0.15, linewidth=0.7)
        plt.plot(epochs_range, val_raw, color="tab:orange", alpha=0.15, linewidth=0.7)
        # Curvas suavizadas (média móvel): o que de fato queremos ler.
        plt.plot(epochs_range, train_smooth, color="tab:blue", linewidth=2, label="Loss (treino)")
        plt.plot(epochs_range, val_smooth, color="tab:orange", linewidth=2, label="Loss (validação)")
        plt.axvline(history["best_epoch"], color="gray", linestyle="--", alpha=0.6,
                    label=f"melhor época (val) = {history['best_epoch']}")
        plt.xlabel("Época")
        plt.ylabel("MSE")
        plt.title(f"Evolução do treinamento — {name}\n(curva suavizada, média móvel de {SMOOTH_WINDOW} épocas)")
        plt.legend()
        plt.tight_layout()
        fname = name.lower().replace(" ", "_").replace("+", "").replace("__", "_")
        fname = "curva_" + fname.strip("_") + ".png"
        plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=120)
        plt.close()

    # ------------------------------------------------------------------
    # Gráfico comparativo: loss de validação de todos os modelos juntos
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for name, history in all_histories.items():
        val_raw = np.array(history["val_loss"])
        val_smooth = moving_average(val_raw)
        epochs_range = np.arange(1, len(val_raw) + 1)
        plt.plot(epochs_range, val_smooth, label=name, linewidth=2)
    plt.xlabel("Época")
    plt.ylabel("MSE (validação, suavizado)")
    plt.title(f"Comparação da evolução do erro de validação entre os modelos\n"
              f"(curvas suavizadas, média móvel de {SMOOTH_WINDOW} épocas)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "comparacao_curvas_validacao.png"), dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Gráfico: função aprendida por cada modelo, sobreposta aos dados
    # (possível pois x é 1-dimensional) - usa o dataset completo (train+
    # val+test) apenas para visualização do ajuste, ordenado por x.
    # ------------------------------------------------------------------
    x_mean, x_std = scaler["x_mean"], scaler["x_std"]
    x_grid_original = np.linspace(0, 10, 400).reshape(-1, 1)
    x_grid_std = (x_grid_original - x_mean) / x_std
    x_grid_tensor = torch.from_numpy(x_grid_std.astype(np.float32))

    full_df = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, (name, model) in enumerate(all_models.items()):
        ax = axes[i]
        ax.scatter(full_df["x"], full_df["y"], s=8, alpha=0.35, color="gray", label="dados")
        model.eval()
        with torch.no_grad():
            y_grid_pred = model(x_grid_tensor).numpy().flatten()
        ax.plot(x_grid_original.flatten(), y_grid_pred, color="crimson", linewidth=2, label="modelo")
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
    for j in range(len(all_models), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "funcoes_aprendidas.png"), dpi=120)
    plt.close()

    print(f"[experiments] Gráficos salvos em: {PLOTS_DIR}")

    return metrics_df, all_histories


if __name__ == "__main__":
    run_all_experiments()
