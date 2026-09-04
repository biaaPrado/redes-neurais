"""
ablation_hparam_search.py
==========================
Antes de comparar "baseline vs. baseline+L1" (etc.), é preciso escolher
UM valor de intensidade para cada técnica (lambda do L1, lambda do L2,
taxa do dropout, coeficiente do momentum). Escolher esses valores
também "no chute" seria injusto - por isso fazemos uma pequena busca
empírica, adicionando CADA técnica isoladamente ao baseline (mesma
arquitetura, mesmo lr, mesmo número de épocas, mesma seed de
inicialização) e variando apenas a intensidade daquela técnica,
escolhendo o valor com menor MSE de validação.

Isso garante que, no estudo de ablação final, cada técnica é comparada
na sua "melhor versão razoável", e não com um valor arbitrário que
poderia sub ou super-estimar seu efeito.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import create_or_load_fixed_split, standardize_to_tensors, RANDOM_SEED
from model import MLP
from trainer import train
from baseline_config import ARCHITECTURE, LEARNING_RATE, EPOCHS, BATCH_SIZE, INIT_SEED

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "logs",
                         "ablation_hparam_search_log.csv")

SEARCH_GRID = {
    "L1":       [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "L2":       [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "Dropout":  [0.1, 0.2, 0.3, 0.4, 0.5],
    "Momentum": [0.5, 0.7, 0.9, 0.95, 0.99],
}


def _train_variant(X_train, y_train, X_val, y_val, technique, value):
    kwargs = dict(momentum=0.0, weight_decay=0.0, l1_lambda=0.0)
    dropout_rate = 0.0
    if technique == "L1":
        kwargs["l1_lambda"] = value
    elif technique == "L2":
        kwargs["weight_decay"] = value
    elif technique == "Dropout":
        dropout_rate = value
    elif technique == "Momentum":
        kwargs["momentum"] = value

    model = MLP(ARCHITECTURE, dropout_rate=dropout_rate, seed=INIT_SEED)
    hist = train(model, X_train, y_train, X_val, y_val,
                 lr=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE,
                 seed=RANDOM_SEED, restore_best=True, **kwargs)
    return hist["best_val_loss"], hist["best_epoch"]


def run_search():
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_projeto1.csv")
    splits = create_or_load_fixed_split(raw_path)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = standardize_to_tensors(
        splits["train"], splits["val"], splits["test"])

    rows = []
    best_per_technique = {}

    for technique, values in SEARCH_GRID.items():
        print(f"--- {technique} ---")
        best_val = float("inf")
        best_value = None
        for value in values:
            val_loss, best_epoch = _train_variant(X_train, y_train, X_val, y_val, technique, value)
            print(f"  valor={value:<8} best_val_loss={val_loss:.4f} @época {best_epoch}")
            rows.append({"tecnica": technique, "valor": value,
                         "best_val_loss": val_loss, "best_epoch": best_epoch})
            if val_loss < best_val:
                best_val = val_loss
                best_value = value
        best_per_technique[technique] = best_value
        print(f"  -> melhor valor para {technique}: {best_value} (val_loss={best_val:.4f})\n")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"[ablation_hparam_search] log salvo em {LOG_PATH}")
    print("Melhores valores escolhidos:", best_per_technique)
    return best_per_technique, df


if __name__ == "__main__":
    run_search()
