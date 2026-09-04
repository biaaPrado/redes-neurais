# Projeto 1 — MLP para Regressão

Este projeto implementa e avalia um Perceptron Multicamadas (MLP) para uma tarefa de regressão, seguindo o roteiro:
  1. Divisão fixa treino/validação/teste (10/10/80);
  2. Desenvolvimento empírico de um baseline "vanilla" (SGD puro, sem truques)
  3. Estudos de ablação avaliando isoladamente L1, L2, Dropout e Momentum sobre o mesmo baseline, sem alterar a 
  arquitetura da rede.

Todo o código usa **Python + PyTorch** (`torch.nn`, `torch.optim.SGD`).
---

## 1. Estrutura do projeto
```
Projeto 1 - Baseline/
├── data/
│   └── dataset_projeto1.csv        # dataset original fornecido
├── src/
│   ├── data_utils.py                # carga do CSV + divisão fixa 10/10/80
│   ├── model.py                     # definição do MLP (nn.Module)
│   ├── trainer.py                   # loop de treino (mini-batch SGD)
│   ├── metrics.py                   # MAE, MSE, RMSE, R²
│   ├── baseline_config.py           # arquitetura/hiperparâmetros finais do baseline
│   ├── baseline_search.py           # busca empírica ampla (arquitetura x lr)
│   ├── ablation_hparam_search.py    # busca da intensidade de cada ablação
│   ├── experiments.py               # treina os 5 modelos finais e gera resultados
│   └── run_all.py                   # roda o pipeline completo, na ordem certa
├── results/
│   ├── splits/                      # train.csv, val.csv, test.csv (divisão FIXA)
│   ├── logs/                        # logs de todas as buscas empíricas realizadas
│   ├── tables/                      # tabela final comparativa de métricas
│   └── plots/                       # todos os gráficos gerados
├── requirements.txt
└── README.md                       
```

### Como executar

```bash
pip install -r requirements.txt
cd src
python run_all.py

```
Isso (re)executa, em ordem, exatamente os passos descritos abaixo. Como a divisão de dados é salva em `results/splits/` 
na primeira execução, rodar o pipeline novamente (mesmo em outra máquina) reproduz a mesma divisão e, graças 
às sementes fixas, os mesmos resultados numéricos.

---

## 2. O dataset

O arquivo `dataset_projeto1.csv` contém 300 amostras com duas colunas:
  - `x`: variável de entrada, contínua, no intervalo `[0, 10]`.
  - `y`: variável-alvo (regressão), contínua, aproximadamente no intervalo `[-1.5, 2.7]`.

**Análise exploratória realizada:**
  - Sem valores nulos e sem linhas duplicadas.
  - `x` é aproximadamente uniforme entre 0 e 10 (média 5.0).
  - Testamos correlação linear de `y` com `x`, `x²`, `sin(x)`, `sqrt(x)` e `log(x+1)` — todas próximas de zero.
  Isso indica que a   relação entre `x` e `y` **não é simples/monotônica**.
  - Ao plotar o diagrama de dispersão, fica claro que `y` segue um padrão **não-linear e não-monotônico** (oscila
  para cima e para baixo ao longo de `x`, com ruído considerável sobreposto) — um cenário tipicamente favorável para
  demonstrar o valor de uma rede MLP (que pode aproximar funções não-lineares arbitrárias) frente a alternativas lineares simples.

Essa não-linearidade motivou a escolha da ativação `tanh` nas camadas ocultas (permite capturar curvas complexas).

---

## 3. Divisão fixa dos dados (10% treino / 10% validação / 80% teste)

Implementada em `src/data_utils.py`, função `create_or_load_fixed_split`.

**Como garantimos que a divisão é sempre a mesma em todas as comparações:**
1. Embaralhamos os 300 índices do dataset com uma semente fixa (`RANDOM_SEED = 42`, `numpy.random.default_rng`).
2. Cortamos os índices embaralhados: os primeiros 10% viram o conjunto de treino (30 amostras), os próximos 10% viram 
validação (30 amostras) e os 80% restantes viram teste (240 amostras).
3. Essa divisão é **salva em disco** (`results/splits/train.csv`, `val.csv`, `test.csv`) na primeira execução.
4. Em qualquer execução futura de qualquer script do projeto, se esses arquivos já existirem, eles são simplesmente **recarregados** — a divisão nunca é gerada de novo. Isso é o que garante, na prática, que o baseline e todas as 4 
ablações sejam sempre avaliados exatamente no mesmo conjunto de teste/validação.

A normalização (padronização) da variável `x` é feita usando média e desvio-padrão calculados **apenas no conjunto de treino**, 
e depois aplicada a validação e teste — evitando vazamento de informação ("data leakage") entre os conjuntos.

> **Por que padronizar `x` e não `y`?** A rede usa `tanh` nas camadas ocultas, que satura (gradiente ≈ 0) para entradas com módulo grande. Sem normalizar, `x` (que vai de 0 a 10) saturaria os primeiros neurônios rapidamente, dificultando o treino com SGD puro (sem otimizador adaptativo). `y` já está numa escala pequena e por isso foi mantida na escala original — o que também torna as métricas finais (MAE, MSE, RMSE) diretamente interpretáveis, sem necessidade de desfazer nenhuma transformação.

---

## 4. Desenvolvimento do baseline (abordagem empírica)

**Regras impostas ao baseline**:
- Otimizador: `torch.optim.SGD` **puro** (`momentum=0`, `weight_decay=0`)
- Sem L1, sem L2, sem dropout.
- Arquitetura "básica": MLP totalmente conectado, ativação `tanh` nas camadas ocultas, saída linear (regressão).

### 4.1 Busca ampla (arquitetura × taxa de aprendizado)

Script: `src/baseline_search.py` → log completo em `results/logs/baseline_search_log.csv`.

Testamos 6 arquiteturas (de 1 e 2 camadas ocultas, 8 a 64 neurônios) × 5 taxas de aprendizado (0.01 a 1.0), por 500 épocas cada, avaliando sempre pelo **MSE de validação** (nunca pelo de treino, que cai artificialmente conforme a rede fica maior/decora os dados).

**Principais observações:**
- `lr ≥ 0.5` **diverge sistematicamente** com `tanh` + inicialização de Xavier, para todas as arquiteturas testadas — o passo de atualização ésimplesmente grande demais.
- Arquiteturas com **2 camadas ocultas** superaram consistentemente as de 1 camada oculta na validação.
- As melhores combinações ficaram entre `lr = 0.05` e `lr = 0.1`.

### 4.2 Refinamento: análise de convergência

Ao observar que a perda de validação dos melhores candidatos ainda estava caindo na época 500 (não havia convergido), estendemos o treino desses candidatos até 3000–8000 épocas para entender a velocidade de convergência.

**Achado importante:** o SGD puro (sem momentum) converge **muito lentamente** neste problema — mesmo com 8000 épocas, a perda 
de validação ainda melhorava lentamente. Isso é *esperado* teoricamente (SGD puro não acelera em regiões de gradiente pequeno/ruidoso) e se tornou um dos pontos centrais de discussão do estudo de ablação de Momentum (seção 6).

Comparamos as arquiteturas `[1, 32, 16, 1]` (609 parâmetros) e `[1, 16, 8, 1]` (177 parâmetros): desempenho de validação praticamente empatado. Por parcimônia (é o que se espera de um baseline "básico") e por reduzir o risco de overfitting bruto 
com apenas 30 exemplos de treino, **escolhemos a arquitetura menor**.

### 4.3 Configuração final do baseline

Definida em `src/baseline_config.py`:

| Hiperparâmetro            | Valor                                                 |
|---------------------------|-------------------------------------------------------|
| Arquitetura               | `[1, 16, 8, 1]` (2 camadas ocultas: 16 e 8 neurônios) |
| Ativação (ocultas)        | `tanh`                                                |
| Ativação (saída)          | linear (regressão)                                    |
| Otimizador                | `SGD` puro (`momentum=0`, `weight_decay=0`)           |
| Taxa de aprendizado (lr)  | 0.1                                                   |
| Épocas                    | 3000                                                  |
| Tamanho do lote (batch)   | 8                                                     |
| Seed de inicialização     | 123 (igual em todos os modelos do estudo)             |

**Escolha do ponto de parada (checkpoint):** ao invés de simplesmente reportar os pesos da última época (que já mostram sinais 
de overfitting — ver gráfico `curva_baseline.png`), guardamos, durante o treino, os pesos correspondentes à **menor perda de validação** observada em qualquer época, e usamos esses pesos para o modelo final. Essa prática ("checkpointing"/early stopping) é aplicada **de forma idêntica a todos os 5 modelos** do projeto (baseline e as 4 ablações), portanto não favorece nenhuma técnica específica — é apenas um critério neutro e padrão para decidir "qual versão do modelo" reportar.

---

## 5. Estudos de ablação

**Regra central:** todos os modelos de ablação usam **exatamente a mesma arquitetura**, os **mesmos pesos iniciais** (mesma seed) e a **mesma taxa de aprendizado/número de épocas/tamanho de lote** do baseline. A única mudança entre eles é ligar **um único** componente extra por vez:

| Modelo                   | L1 | L2 | Dropout | Momentum |
|--------------------------|----|----|---------|----------|
| Baseline                 | -  | -  | -       | -        |
| Baseline + L1            | ✔ | -   | -       | -        |
| Baseline + L2            | -  | ✔  | -       | -        |
| Baseline + Dropout       | -  | -  | ✔       | -        |
| Baseline + Momentum      | -  | -  | -       | ✔        |

Como cada componente foi implementado **sem alterar a arquitetura** (ver comentários em `src/model.py` e `src/trainer.py`):

- **L1**: penalidade `l1_lambda * Σ|w|` somada manualmente à função de perda antes do `.backward()` (PyTorch não tem um parâmetro pronto para L1 no otimizador, ao contrário do L2). 
- **L2**: passado como `weight_decay` para `torch.optim.SGD` — o PyTorch soma `weight_decay * w` ao gradiente antes do passo de atualização, equivalente a uma penalidade `(λ/2)·‖w‖²` na perda.
- **Dropout**: a classe `MLP` sempre tem uma camada `nn.Dropout(p)` após cada ativação oculta; com `p=0`, ela é matematicamente a função identidade (não afeta o baseline). Só na ablação de dropout `p>0`.
- **Momentum**: passado como parâmetro `momentum` do `torch.optim.SGD` — com `momentum=0` (baseline) o otimizador se comporta como SGD puro.

### 5.1 Escolha da intensidade de cada técnica

Antes de comparar "baseline vs. baseline+técnica", era preciso escolher **um valor razoável** para cada intensidade (λ do L1, λ do L2, taxa do dropout, coeficiente do momentum) — escolher esses valores "no chute" seria injusto com a técnica. Fizemos uma pequena busca empírica (`src/ablation_hparam_search.py`, log em `results/logs/ablation_hparam_search_log.csv`), testando 5 valores por técnica e escolhendo o de menor MSE de validação:

| Técnica   | Valores testados                       | Melhor valor | MSE val. no melhor valor |
|-----------|----------------------------------------|--------------|--------------------------|
| L1        | 0.0001, 0.0005, 0.001, 0.005, 0.01     | **0.0001**   | 0.4754                   |
| L2        | 0.0001, 0.0005, 0.001, 0.005, 0.01     | **0.0001**   | 0.4763                   |
| Dropout   | 0.1, 0.2, 0.3, 0.4, 0.5                | **0.1**      | 0.5903                   |
| Momentum  | 0.5, 0.7, 0.9, 0.95, 0.99              | **0.7**      | 0.4369                   |

**Observações desta busca, já bastante reveladoras:**
- Para **L1 e L2**, valores de penalidade mais altos que `0.0001` **pioram** progressivamente o desempenho de validação. Com apenas 30 exemplos de treino e uma rede pequena (177 parâmetros), o modelo já não tem tanta capacidade "sobrando" para regularizar agressivamente — o ponto ótimo de regularização é bem sutil.

- Para **dropout**, o mesmo padrão, ainda mais acentuado: mesmo `p=0.1` (o mais leve testado) já piora o MSE de validação em relação ao baseline (0.59 vs. 0.48). Isso faz sentido: a rede já é pequena (16 e 8 neurônios), então "apagar" neurônios aleatoriamente durante o treino reduz demais a capacidade efetiva do modelo, prejudicando o aprendizado ao invés de ajudar a generalizar.

- Para **momentum**, o efeito é bem diferente dos anteriores: não é uma regularização, e sim uma aceleração da otimização. Como vimos que o SGD puro converge lentamente (seção 4.2), momentum (0.5–0.7) permite que a rede atinja uma perda de validação menor **dentro do mesmo orçamento de 3000 épocas** — não por "regularizar", mas por otimizar melhor a mesma função de perda. Valores muito altos (0.95, 0.99), porém, pioram o resultado (oscilação/instabilidade), como esperado.

Esses valores (L1=0.0001, L2=0.0001, Dropout=0.1, Momentum=0.7) foram os usados no estudo de ablação final (`src/experiments.py`).

---

## 6. Resultados finais

Script: `src/experiments.py`. Tabela completa em `results/tables/metricas_comparativas.csv`. Todas as métricas abaixo são
calculadas com o modelo no seu **melhor checkpoint de validação** (ver coluna "melhor época").

### 6.1 Tabela comparativa completa

| modelo | conjunto | MAE | MSE | RMSE | R² | melhor época |
|---|---|---|---|---|---|---|
| Baseline | treino | 0.3191 | 0.1518 | 0.3896 | 0.5885 | 2701 |
| Baseline | validação | 0.5485 | 0.4775 | 0.6910 | 0.3766 | 2701 |
| **Baseline** | **teste** | **0.4887** | **0.3816** | **0.6178** | **0.2597** | 2701 |
| Baseline + L1 | treino | 0.3208 | 0.1531 | 0.3912 | 0.5849 | 2628 |
| Baseline + L1 | validação | 0.5291 | 0.4754 | 0.6895 | 0.3794 | 2628 |
| **Baseline + L1** | **teste** | **0.5056** | **0.4120** | **0.6419** | **0.2008** | 2628 |
| Baseline + L2 | treino | 0.3260 | 0.1563 | 0.3954 | 0.5761 | 2441 |
| Baseline + L2 | validação | 0.5395 | 0.4763 | 0.6901 | 0.3783 | 2441 |
| **Baseline + L2** | **teste** | **0.4966** | **0.3967** | **0.6298** | **0.2305** | 2441 |
| Baseline + Dropout | treino | 0.3968 | 0.2279 | 0.4774 | 0.3819 | 2811 |
| Baseline + Dropout | validação | 0.6434 | 0.5903 | 0.7683 | 0.2294 | 2811 |
| **Baseline + Dropout** | **teste** | **0.5740** | **0.5088** | **0.7133** | **0.0131** | 2811 |
| Baseline + Momentum | treino | 0.2756 | 0.1285 | 0.3585 | 0.6514 | 2050 |
| Baseline + Momentum | validação | 0.5139 | 0.4369 | 0.6610 | 0.4297 | 2050 |
| **Baseline + Momentum** | **teste** | **0.5013** | **0.4170** | **0.6458** | **0.1910** | 2050 |

### 6.2 Gráficos gerados (pasta `results/plots/`)

- `curva_baseline.png`, `curva_baseline_l1.png`, `curva_baseline_l2.png`, `curva_baseline_dropout.png`, `curva_baseline_momentum.png` — evolução de treino x validação (MSE por época) de cada modelo individualmente.
- `comparacao_curvas_validacao.png` — as 5 curvas de validação sobrepostas, para comparar diretamente a velocidade/qualidade de
  convergência.
- `funcoes_aprendidas.png` — a função aprendida por cada modelo (linha), sobreposta aos dados reais (pontos), possível pois `x` é
  1-dimensional. Ótimo para "ver" visualmente o que cada técnica fez com o ajuste.

### 6.3 Discussão dos resultados

- **Momentum foi a técnica com melhor desempenho em validação** (MSE 0.4369, R²=0.43) e o menor MAE/MSE de treino também — resultado esperado, já que, como discutido na seção 4.2, o SGD puro converge lentamente neste problema; momentum simplesmente permite explorar melhor a superfície de perda dentro do mesmo orçamento de épocas.

- **L1 e L2, nas intensidades ótimas encontradas, tiveram desempenho de validação muito próximo do baseline** (diferenças de MSE < 0.001) —  ou seja, praticamente não mudaram o comportamento do modelo. Isso é coerente com a análise da seção 5.1: a rede já é pequena e o conjunto de treino já é escasso (30 amostras), então não há muito "excesso de capacidade" para essas penalidades cortarem.

- **Dropout piorou claramente o desempenho** em todos os conjuntos (MSE de validação 0.59 vs. 0.48 do baseline; R² de teste caiu de 0.26 para praticamente 0.01). Ao observar `funcoes_aprendidas.png`, fica visível que o modelo com dropout aprendeu uma curva mais "achatada", perdendo parte da segunda oscilação do padrão real — sinal de que o dropout, nesta rede pequena com poucos dados, **reduziu a capacidade efetiva** da rede a ponto de prejudicar o ajuste, ao invés de apenas combater overfitting.

- **Um ponto de honestidade estatística importante**: no conjunto de **validação** (30 amostras), a ordem de desempenho foi Momentum > L1 ≈ L2 ≈ Baseline > Dropout; mas no conjunto de **teste** (240 amostras), o **Baseline puro teve o melhor R²** (0.26), superando inclusive o Momentum (0.19). Isso acontece porque o conjunto de validação, sendo muito pequeno (apenas 10% dos dados, 30 pontos), tem alta variância — a "melhor época" e o "melhor hiperparâmetro" escolhidos por ele nem sempre generalizam perfeitamente para uma amostra de teste 8× maior. Esse é um efeito colateral esperado de se usar frações de treino/validação tão pequenas (10%/10%), e reforça a importância de reportar métricas no conjunto de teste (não só na validação) antes de tirar conclusões definitivas sobre qual técnica é "melhor".

- Em todos os modelos, o **erro de treino é substancialmente menor que o de validação/teste** (ex.: baseline R²=0.59 no treino vs. 0.26 no teste) — um sinal claro de overfitting, esperado dado que há apenas 30 exemplos de treino para uma função não-linear com múltiplas oscilações. Isso também explica por que as curvas de validação em `comparacao_curvas_validacao.png` são tão ruidosas: o MSE de validação é calculado sobre apenas 30 pontos, então tem variância alta de época para época.

### 6.4 Limitações e possíveis extensões

- O conjunto de treino de apenas 30 amostras é pequeno para a complexidade da função-alvo (múltiplas oscilações não-lineares),
  limitando o quanto qualquer técnica de regularização pode ajudar sem mais dados. 
- Não foi feita busca de intensidade combinando duas técnicas ao mesmo tempo (ex.: L2 + Momentum), pois o enunciado pede avaliação isolada de cada componente sobre o baseline.
- Uma extensão natural seria repetir todo o estudo com múltiplas sementes de inicialização e reportar médias ± desvio-padrão das
  métricas, para quantificar a variância introduzida pela inicialização aleatória dos pesos — o que não foi feito aqui para manter o escopo dentro do que foi pedido (comparação com pesos iniciais idênticos entre os modelos).

---

## 7. Requisitos

```
torch
numpy
pandas
matplotlib
```

(ver `requirements.txt`)
