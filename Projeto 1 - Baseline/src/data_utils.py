"""
data_utils.py
=============
Responsável por:
  1) Carregar o dataset bruto (CSV com colunas 'x' e 'y').
  2) Gerar UMA ÚNICA VEZ a divisão treino/validação/teste (10%/10%/80%), usando uma semente (seed) fixa.
  3) Persistir essa divisão em disco (results/splits/*.csv), de forma que TODOS os scripts do projeto (busca do baseline, treino 
  do baseline, estudos de ablação) leiam sempre os mesmos três arquivos.
  4) Converter os dados para tensores PyTorch, prontos para treino.

Por que isso importa?
----------------------
Se cada script gerasse sua própria divisão aleatória, os modelos estariam sendo comparados em conjuntos diferentes e qualquer 
diferença de desempenho poderia vir da divisão dos dados, e não da técnica testada (L1, L2, dropout, momentum) - o que invalidaria
o estudo de ablação.

Estratégia usada:
------------------
- Embaralhamos os índices do dataset com uma semente fixa.
- Cortamos os índices embaralhados em 3 blocos: 10% treino, 10% validação, 80% teste.
- Salvamos os três subconjuntos em disco na primeira execução. Nas execuções seguintes, se os arquivos já existirem, eles são 
apenas recarregados (reprodutibilidade garantida, mesmo em dias/execuções diferentes).
- Padronizamos (normalizamos) a variável de entrada X com média/desvio calculados SOMENTE no conjunto de treino, para não vazar
informação de validação/teste para o pré-processamento ("data leakage").
"""

import os
import numpy as np
import pandas as pd
import torch

RANDOM_SEED = 42          # semente fixa para reprodutibilidade de TODO o projeto
TRAIN_FRAC = 0.10         # fração do dataset usada para treino (10%)
VAL_FRAC = 0.10           # fração do dataset usada para validação (10%)
TEST_FRAC = 0.80          # fração do dataset usada para teste (80%)

SPLIT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "splits") # diretório para salvar as divisões fixas


def create_or_load_fixed_split(raw_csv_path, split_dir=SPLIT_DIR, seed=RANDOM_SEED):
    """
    Garante que existe (criando se necessário) uma divisão fixa do dataset em treino/validação/teste, e a retorna sempre da mesma forma.

    Retorna
    -------
    dict com chaves 'train', 'val', 'test', cada uma um DataFrame com colunas ['x', 'y'].
    """
    os.makedirs(split_dir, exist_ok=True)
    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")
    test_path = os.path.join(split_dir, "test.csv")

    # Se a divisão já foi gerada antes, apenas recarregamos -> garante que a MESMA divisão seja usada em toda e qualquer execução futura.
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print("[data_utils] Divisão fixa já existente carregada de disco " f"({split_dir}).")
        return {"train": train_df, "val": val_df, "test": test_df}

    # --- Caso contrário, geramos a divisão pela primeira (e única) vez ---
    df = pd.read_csv(raw_csv_path)
    n = len(df)

    rng = np.random.default_rng(seed)           # gerador de aleatoriedade com seed fixa
    shuffled_idx = rng.permutation(n)           # embaralha os índices 0..n-1

    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    n_test = n - n_train - n_val                # resto, para garantir soma == n mesmo com arredondamento

    train_idx = shuffled_idx[:n_train]
    val_idx = shuffled_idx[n_train:n_train + n_val]
    test_idx = shuffled_idx[n_train + n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[data_utils] Nova divisão fixa criada e salva em '{split_dir}':")
    print(f"treino={len(train_df)} ({len(train_df)/n:.1%}) | "
          f"validação={len(val_df)} ({len(val_df)/n:.1%}) | "
          f"teste={len(test_df)} ({len(test_df)/n:.1%})")

    return {"train": train_df, "val": val_df, "test": test_df}


def standardize_to_tensors(train_df, val_df, test_df, feature_col="x", target_col="y"):
    """
    Padroniza a coluna de entrada usando média/desvio do TREINO e retorna tudo já como tensores PyTorch (dtype float32), 
    prontos para uso em nn.Module.

    Por que padronizar X?
    ----------------------
    A rede usa tanh nas camadas ocultas, que satura (gradiente ~0) para entradas com módulo grande. Nosso x original vai 
    de 0 a 10; sem normalizar, os neurônios da primeira camada saturariam facilmente, dificultando o aprendizado com SGD
    puro (sem otimizador adaptativo como Adam, que corrigiria isso sozinho). Padronizar para média 0 e desvio 1 mantém as
    ativações numa faixa numérica saudável para tanh.

    Por que NÃO padronizar y?
    ---------------------------
    y já está numa escala pequena (aprox. -1.5 a 2.7), então não é estritamente necessário, e manter y na escala original 
    torna as métricas (MAE, MSE, RMSE, R²) diretamente interpretáveis.
    """
    x_mean = train_df[feature_col].mean()   # média do treino para padronização
    x_std = train_df[feature_col].std()     # desvio padrão do treino para padronização

    #conversão dos dados de array (usados no Numpy para tratar de médias e desvios) para tensores visto que o PyTorch
    #consegue rastrear de forma mais eficiente as operações feitas anteriormente, e assim calcular o gradiente automaticamente
    #durante o treino, importante para o backpropagation. 
    def _prep(df):
        X = ((df[feature_col].values - x_mean) / x_std).reshape(-1, 1).astype(np.float32)  # padroniza e transforma em array 2D
        y = df[target_col].values.reshape(-1, 1).astype(np.float32)   # transforma em array 2D
        return torch.from_numpy(X), torch.from_numpy(y)              # converte para tensores PyTorch

    X_train, y_train = _prep(train_df)     # converte treino para tensores
    X_val, y_val = _prep(val_df)           # converte validação para tensores
    X_test, y_test = _prep(test_df)        # converte teste para tensores

    scaler_params = {"x_mean": float(x_mean), "x_std": float(x_std)}     # salva os parâmetros de padronização para uso posterior (ex.: na hora de fazer previsões em novos dados)
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_params 


if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_projeto1.csv")  # caminho para o dataset bruto
    splits = create_or_load_fixed_split(raw_path)       # cria ou carrega a divisão fixa
    for name, d in splits.items():                      # imprime o tamanho de cada subconjunto
        print(name, d.shape)
