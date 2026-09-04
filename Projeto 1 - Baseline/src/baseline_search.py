"""
baseline_search.py
===================
Implementa a "abordagem empírica" pedida no enunciado: treina várias
combinações de arquitetura (número/tamanho de camadas ocultas) e taxa de
aprendizado, TODAS usando SGD puro (momentum=0, weight_decay=0,
l1_lambda=0, dropout_rate=0), e escolhe como baseline a combinação com
MENOR erro (MSE) no conjunto de VALIDAÇÃO.

Cada combinação testada é registrada em
results/logs/baseline_search_log.csv, documentando todas as análises
feitas durante o desenvolvimento do baseline (conforme exigido no
enunciado).

Por que MSE de validação como critério de seleção (e não o de treino)?
--------------------------------------------------------------------------
O erro de treino tende a cair conforme a rede fica mais "poderosa" (mais
neurônios), mesmo que ela esteja apenas decorando os 30 exemplos de
treino (overfitting). O conjunto de validação, que o modelo nunca vê
durante o ajuste dos pesos, é o que revela se o modelo está de fato
generalizando - por isso é o critério correto para escolher a
arquitetura/hiperparâmetros do baseline.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import create_or_load_fixed_split, standardize_to_tensors, RANDOM_SEED
from model import MLP
from trainer import train

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LOG_PATH = os.path.join(RESULTS_DIR, "logs", "baseline_search_log.csv")

# ----------------------------------------------------------------------
# ESPAÇO DE BUSCA (empírico): variamos arquitetura e taxa de aprendizado.
# Mantemos batch_size e número de épocas fixos nesta etapa, para isolar
# o efeito desses dois fatores (arquitetura x lr).
# ----------------------------------------------------------------------
ARCHITECTURES = {
    "1x8":     [1, 8, 1],
    "1x16":    [1, 16, 1],
    "1x32":    [1, 32, 1],
    "1x64":    [1, 64, 1],
    "2x16-8":  [1, 16, 8, 1],
    "2x32-16": [1, 32, 16, 1],
}
LEARNING_RATES = [1.0, 0.5, 0.1, 0.05, 0.01]
SEARCH_EPOCHS = 500
BATCH_SIZE = 8
INIT_SEED = 123   # mesma seed de inicialização de pesos em toda a busca,
                   # para que a comparação entre arquiteturas/lr não seja
                   # "poluída" por sorte na inicialização.


def run_search():
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_projeto1.csv")
    splits = create_or_load_fixed_split(raw_path)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = standardize_to_tensors(
        splits["train"], splits["val"], splits["test"])

    results = []
    print("=== Busca empírica do baseline (SGD puro, sem regularização) ===\n")

    for arch_name, layer_sizes in ARCHITECTURES.items():
        for lr in LEARNING_RATES:
            model = MLP(layer_sizes, dropout_rate=0.0, seed=INIT_SEED)
            history = train(model, X_train, y_train, X_val, y_val,
                             lr=lr, epochs=SEARCH_EPOCHS, batch_size=BATCH_SIZE,
                             momentum=0.0, weight_decay=0.0, l1_lambda=0.0,
                             seed=RANDOM_SEED, verbose=False)

            final_train_loss = history["train_loss"][-1]
            final_val_loss = history["val_loss"][-1]
            best_val_loss = float(np.min(history["val_loss"]))
            best_val_epoch = int(np.argmin(history["val_loss"])) + 1
            diverged = not np.isfinite(final_val_loss)

            n_params = model.n_parameters()

            results.append({
                "arquitetura": arch_name,
                "layer_sizes": str(layer_sizes),
                "n_parametros": n_params,
                "learning_rate": lr,
                "epochs": SEARCH_EPOCHS,
                "batch_size": BATCH_SIZE,
                "loss_treino_final": final_train_loss,
                "loss_val_final": final_val_loss,
                "loss_val_minima": best_val_loss,
                "epoca_melhor_val": best_val_epoch,
                "divergiu": diverged,
            })

            status = ("DIVERGIU" if diverged else
                      f"val_final={final_val_loss:.4f} (melhor val={best_val_loss:.4f} @ep{best_val_epoch})")
            print(f"  arch={arch_name:9s} lr={lr:<5} -> {status}")

    log_df = pd.DataFrame(results).sort_values("loss_val_minima")
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_df.to_csv(LOG_PATH, index=False)

    print(f"\n[baseline_search] Log completo salvo em: {LOG_PATH}")

    valid = log_df[~log_df["divergiu"]]
    best = valid.iloc[0]
    print("\n=== Melhor configuração encontrada (candidato a baseline) ===")
    print(best.to_string())

    return best, log_df


if __name__ == "__main__":
    run_search()
