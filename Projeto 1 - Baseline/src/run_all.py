"""
run_all.py
==========
Executa o pipeline completo do projeto, na ordem correta, do zero:

  1) Cria (ou carrega, se já existir) a divisão fixa treino/val/teste.
  2) Roda a busca empírica ampla de arquitetura x taxa de aprendizado
     (gera results/logs/baseline_search_log.csv).
  3) Roda a busca de intensidade de cada técnica de ablação (gera
     results/logs/ablation_hparam_search_log.csv).
  4) Treina os 5 modelos finais (baseline + 4 ablações), gera todos os
     gráficos e a tabela comparativa final de métricas.

Basta rodar:  python run_all.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import create_or_load_fixed_split

import baseline_search
import ablation_hparam_search
import experiments


def main():
    print("\n########## ETAPA 1/4: divisão fixa dos dados ##########")
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_projeto1.csv")
    create_or_load_fixed_split(raw_path)

    print("\n########## ETAPA 2/4: busca empírica do baseline ##########")
    baseline_search.run_search()

    print("\n########## ETAPA 3/4: busca de intensidade das ablações ##########")
    ablation_hparam_search.run_search()

    print("\n########## ETAPA 4/4: treino final (baseline + 4 ablações) ##########")
    experiments.run_all_experiments()

    print("\nConcluído! Veja os resultados em ../results/")


if __name__ == "__main__":
    main()
