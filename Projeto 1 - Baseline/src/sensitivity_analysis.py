"""
sensitivity_analysis.py
=========================
Testa a sensibilidade do modelo a 3 hiperparâmetros de treino: learning rate, dropout rate e batch size — variando CADA UM isoladamente.

Diferente das buscas empíricas já feitas em baseline_search.py e ablation_hparam_search.py (cujo objetivo era ENCONTRAR o melhor 
valor de cada hiperparâmetro), este script serve para MOSTRAR o que acontece quando nos afastamos do valor já escolhido — usando
sempre a arquitetura final e o orçamento completo de 3000 épocas (o mesmo do treino oficial).

Duas referências, não uma só - e por quê
-------------------------------------------
Em vez de usar um único "modelo combinado" (lr=0.1 + dropout=0.1 + batch=8, que nunca existiu como modelo oficial do projeto), ancoramos
cada grupo de teste no modelo OFICIAL mais relevante para aquele hiperparâmetro:

  - Testes de LEARNING RATE e BATCH SIZE usam dropout=0 como referência 
    -> é exatamente o Baseline oficial (o modelo "carro-chefe" do projeto, em experiments.py). Faz sentido testar sensibilidade de
       lr/batch nesse modelo, não num modelo que já sabemos ser pior (o de dropout).

  - Testes de DROPOUT RATE usam lr=0.1/batch=8 como referência (os valores oficiais do baseline_config.py) 
    -> é exatamente a "Baseline + Dropout" oficial (a ablação já feita em experiments.py).

    Grupo LEARNING RATE (dropout=0 fixo): lr=0.01 | lr=0.1 (referência = Baseline oficial) | lr=0.3

    Grupo BATCH SIZE (dropout=0 fixo): batch=1 | batch=8 (referência = Baseline oficial) | batch=30

    Grupo DROPOUT RATE (lr=0.1, batch=8 fixos): dropout=0.05 | dropout=0.1 (referência = Baseline+Dropout oficial) | dropout=0.3

Onde os resultados são salvos
-------------------------------
results/analises_extra/sensibilidade/plots/   -> 1 gráfico por grupo (lr, dropout, batch)
results/analises_extra/sensibilidade/tables/  -> tabela única com todos os testes

Cada gráfico/tabela é salvo IMEDIATAMENTE após ser gerado. Nada aqui sobrescreve resultados de experiments.py ou de outras análises extras.
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

BASE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SENS_DIR = os.path.join(BASE_RESULTS_DIR, "analises_extra", "sensibilidade")
SENS_PLOTS_DIR = os.path.join(SENS_DIR, "plots")
SENS_TABLES_DIR = os.path.join(SENS_DIR, "tables")

SMOOTH_WINDOW = 51
DROPOUT_RATE_OFICIAL = 0.1  # valor escolhido em ablation_hparam_search.py / usado em experiments.py

# ----------------------------------------------------------------------
# Definição dos testes, agrupados por hiperparâmetro. Cada grupo tem seu próprio "valor de referência" (marcado abaixo), que
# corresponde a um modelo oficial já existente no projeto.
# ----------------------------------------------------------------------
LR_TESTS = {
    "lr=0.01":             dict(lr=0.01, dropout_rate=0.0, batch_size=BATCH_SIZE),
    "lr=0.1 (Baseline)":   dict(lr=LEARNING_RATE, dropout_rate=0.0, batch_size=BATCH_SIZE),  # referência
    "lr=0.3":              dict(lr=0.3, dropout_rate=0.0, batch_size=BATCH_SIZE),
}
LR_REFERENCE = "lr=0.1 (Baseline)"

BATCH_TESTS = {
    "batch=1":               dict(lr=LEARNING_RATE, dropout_rate=0.0, batch_size=1),
    "batch=8 (Baseline)":    dict(lr=LEARNING_RATE, dropout_rate=0.0, batch_size=BATCH_SIZE),  # referência
    "batch=30":              dict(lr=LEARNING_RATE, dropout_rate=0.0, batch_size=30),
}
BATCH_REFERENCE = "batch=8 (Baseline)"

DROPOUT_TESTS = {
    "dropout=0.05":                    dict(lr=LEARNING_RATE, dropout_rate=0.05, batch_size=BATCH_SIZE),
    "dropout=0.1 (Baseline+Dropout)":  dict(lr=LEARNING_RATE, dropout_rate=DROPOUT_RATE_OFICIAL, batch_size=BATCH_SIZE),  # referência
    "dropout=0.3":                     dict(lr=LEARNING_RATE, dropout_rate=0.3, batch_size=BATCH_SIZE),
}
DROPOUT_REFERENCE = "dropout=0.1 (Baseline+Dropout)"

GROUPS = {
    "Learning Rate": (LR_TESTS, LR_REFERENCE),
    "Batch Size":    (BATCH_TESTS, BATCH_REFERENCE),
    "Dropout Rate":  (DROPOUT_TESTS, DROPOUT_REFERENCE),
}


def moving_average(values, window=SMOOTH_WINDOW): #suaviza a curva sem alterar os valores da tabela final 
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    weight_used = np.convolve(np.ones_like(values), kernel, mode="same")
    return smoothed / weight_used


def _ensure_dirs():
    os.makedirs(SENS_PLOTS_DIR, exist_ok=True)
    os.makedirs(SENS_TABLES_DIR, exist_ok=True)


def _load_data(): #carrega os dados e padroniza para PyTorch 
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_projeto1.csv")
    splits = create_or_load_fixed_split(raw_path)
    return standardize_to_tensors(splits["train"], splits["val"], splits["test"])


def _train_test(lr, dropout_rate, batch_size, X_train, y_train, X_val, y_val): #faz o treinamento, com o mesmo modelo variando só os hiperparametros 
    model = MLP(ARCHITECTURE, dropout_rate=dropout_rate, seed=INIT_SEED)
    history = train(model, X_train, y_train, X_val, y_val,
                     lr=lr, epochs=EPOCHS, batch_size=batch_size,
                     momentum=0.0, weight_decay=0.0, l1_lambda=0.0,
                     seed=RANDOM_SEED, restore_best=True)
    return model, history


def run_group(group_name, tests, reference_key, X_train, y_train, X_val, y_val, X_test, y_test,
              cache):
    """Treina todos os testes de um grupo (reaproveitando do `cache` um
    teste já treinado em outro grupo, caso a configuração seja idêntica —
    evita retreinar a mesma coisa duas vezes)."""
    print(f"\n=== Grupo: {group_name} ===")
    histories, rows = {}, []

    for name, cfg in tests.items():
        cache_key = (cfg["lr"], cfg["dropout_rate"], cfg["batch_size"])
        if cache_key in cache:
            print(f"  {name}: já treinado antes (reaproveitando) — "
                  f"lr={cfg['lr']}, dropout={cfg['dropout_rate']}, batch={cfg['batch_size']}")
            model, history = cache[cache_key]
        else:
            print(f"  treinando {name}: lr={cfg['lr']}, dropout={cfg['dropout_rate']}, " f"batch={cfg['batch_size']}")
            model, history = _train_test(cfg["lr"], cfg["dropout_rate"], cfg["batch_size"], X_train, y_train, X_val, y_val)
            cache[cache_key] = (model, history)

        histories[name] = history
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
        m = all_metrics(y_test, y_pred_test)
        m.update({
            "grupo": group_name, "teste": name,
            "learning_rate": cfg["lr"], "dropout_rate": cfg["dropout_rate"],
            "batch_size": cfg["batch_size"],
            "eh_referencia": (name == reference_key),
            "melhor_epoca": history["best_epoch"],
            "melhor_val_loss": history["best_val_loss"],
        })
        rows.append(m)
        print(f"    melhor época={history['best_epoch']} | val_loss={history['best_val_loss']:.4f} "
              f"| teste: MAE={m['MAE']:.4f} R2={m['R2']:.4f}")

    # --- gráfico do grupo, salvo imediatamente ---
    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        val_smooth = moving_average(history["val_loss"])
        is_ref = (name == reference_key)
        plt.plot(np.arange(1, len(val_smooth) + 1), val_smooth, "-" if is_ref else "--", linewidth=2.5 if is_ref else 1.8, label=name)
    plt.xlabel("Época")
    plt.ylabel("MSE (validação, suavizado)")
    plt.title(f"Sensibilidade — {group_name}\n" f"(curva suavizada, média móvel de {SMOOTH_WINDOW} épocas)")
    plt.legend()
    plt.tight_layout()
    fname = "sensibilidade_" + group_name.lower().replace(" ", "_") + ".png"
    plot_path = os.path.join(SENS_PLOTS_DIR, fname)
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  [salvo] {plot_path}")

    return rows


def main():
    _ensure_dirs()
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = _load_data()

    cache = {}  # evita retreinar a mesma config (ex.: o Baseline oficial é usado em 2 grupos)
    all_rows = []
    for group_name, (tests, reference_key) in GROUPS.items():
        rows = run_group(group_name, tests, reference_key, X_train, y_train, X_val, y_val, X_test, y_test, cache)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    table_path = os.path.join(SENS_TABLES_DIR, "sensibilidade_resultados.csv")
    df.to_csv(table_path, index=False)
    print(f"\n[salvo] {table_path}")

    print(f"\nConcluído! Veja os resultados em {SENS_DIR}")
    return df


if __name__ == "__main__":
    main()