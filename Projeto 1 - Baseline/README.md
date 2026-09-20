# Projeto 1 — Regressão com MLP e Estudo de Ablação

Projeto da disciplina **Introdução às Redes Neurais Artificiais**, implementando um Perceptron Multicamadas (MLP) em **PyTorch** para uma tarefa de regressão, com:

- Divisão fixa dos dados em treino (10%), validação (10%) e teste (80%)
- Desenvolvimento empírico de um baseline "vanilla" (SGD puro, sem regularização)
- Estudo de ablação avaliando isoladamente L1, L2, Dropout e Momentum
- Análise de sensibilidade a hiperparâmetros (learning rate, dropout, batch size)

**Autora:** Beatriz Prado Soche (RA: 176112)

---

## Estrutura do repositório

```
Projeto 1 - Baseline/
├── data/
│   └── dataset_projeto1.csv         # dataset fornecido
├── src/
│   ├── data_utils.py                 # carga do CSV + divisão fixa 10/10/80
│   ├── model.py                      # definição do MLP (nn.Module)
│   ├── trainer.py                    # loop de treino (mini-batch SGD)
│   ├── metrics.py                    # MAE, MSE, RMSE, R²
│   ├── baseline_config.py            # arquitetura/hiperparâmetros do baseline
│   ├── baseline_search.py            # busca empírica ampla (arquitetura x lr)
│   ├── ablation_hparam_search.py     # busca da intensidade de cada ablação
│   ├── experiments.py                # treina os 5 modelos e gera resultados
│   ├── sensitivity_analysis.py       # análise de sensibilidade (lr, dropout, batch)
│   └── run_all.py                    # executa todo o pipeline em sequência
├── results/
│   ├── analises_extras/sensibilidade
    │   ├── tables/                   # métricas comparativas finais
    │   └── plots/                    # gráficos gerados                     
│   ├── splits/                       # divisão fixa treino/val/teste
│   ├── logs/                         # logs das buscas empíricas
│   ├── tables/                       # métricas comparativas finais
│   └── plots/                        # gráficos gerados
├── relatorio.tex                     # relatório em LaTeX
├── Relatório Projeto                 # relatório em PDF
├── requirements.txt
└── README.md
```

---

## Requisitos

- Python 3.10+
- Dependências listadas em `requirements.txt`:

```
torch
numpy
pandas
matplotlib
```

## Instalação

```bash
git clone https://github.com/biaaPrado/redes-neurais.git
cd "redes-neurais/Projeto 1 - Baseline"
pip install -r requirements.txt
```

## Como executar

Rodar o pipeline completo (divisão dos dados, busca de hiperparâmetros, treino dos 5 modelos e geração de gráficos/tabelas):

```bash
cd src
python run_all.py
```

Também é possível rodar cada etapa separadamente:

```bash
python data_utils.py               # gera a divisão fixa dos dados
python baseline_search.py          # busca empírica de arquitetura/lr
python ablation_hparam_search.py   # busca de intensidade das ablações
python experiments.py              # treina baseline + 4 ablações
python sensitivity_analysis.py     # análise de sensibilidade a hiperparâmetros
```

Os resultados (tabelas, logs e gráficos) são salvos automaticamente em `results/`.

## Relatório

O relatório completo do projeto (metodologia, resultados e discussão) está em [`relatorio.tex`](./relatorio.tex) / `relatorio.pdf`.

## Reprodutibilidade

Todas as etapas usam sementes aleatórias fixas (divisão dos dados e inicialização dos pesos), e a divisão treino/validação/teste é persistida em `results/splits/` na primeira execução, garantindo que todos os modelos sejam comparados sobre exatamente os mesmos dados.