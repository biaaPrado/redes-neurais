"""
baseline_config.py
====================
Centraliza a configuração final do baseline, escolhida por meio da busca
empírica documentada em:
    - results/logs/baseline_search_log.csv   (busca ampla: arquitetura x lr)
    - análise de convergência estendida (ver README.md, seção "Análises
      realizadas durante o desenvolvimento do baseline")

Por que centralizar aqui?
---------------------------
Tanto o script que treina o baseline final quanto o script que faz a
busca de intensidade das ablações (ablation_hparam_search.py) e o script
que roda todos os experimentos finais (experiments.py) precisam usar
EXATAMENTE a mesma arquitetura, taxa de aprendizado, número de épocas,
tamanho de lote e semente de inicialização. Definir essas constantes em
um único módulo evita que algum script "esqueça" de replicar um valor e
acabe comparando modelos de forma inconsistente.

Resumo da busca (ver README.md para a análise completa):
------------------------------------------------------------
- Testamos arquiteturas de 1 e 2 camadas ocultas (8 a 64 neurônios) e
  taxas de aprendizado de 0.01 a 1.0, com SGD puro, por 500 épocas
  (busca ampla). lr >= 0.5 sistematicamente diverge nesta rede com
  ativação tanh; arquiteturas com 2 camadas ocultas superaram as de 1
  camada na validação.
- Refinamos os 2 melhores candidatos treinando por mais épocas (até
  8000) para entender a velocidade de convergência do SGD puro (que é
  lenta, como esperado - sem momentum, sem taxa adaptativa). A
  arquitetura [1, 16, 8, 1] com lr=0.1 atingiu desempenho de validação
  praticamente igual ao da arquitetura maior [1, 32, 16, 1], mas com
  menos de 1/3 dos parâmetros (177 vs. 609) - preferimos a menor por
  ser mais "básica"/parcimoniosa, com risco menor de overfitting bruto.
- Fixamos o orçamento de treino em 3000 épocas (equilíbrio entre tempo
  de execução e convergência) e usamos "checkpointing" pelo menor erro
  de validação (ver trainer.py) para reportar o modelo no seu melhor
  ponto, e não necessariamente na última época.
"""

ARCHITECTURE = [1, 16, 8, 1]   # 1 entrada -> 16 neurônios -> 8 neurônios -> 1 saída
LEARNING_RATE = 0.1
EPOCHS = 3000
BATCH_SIZE = 8
INIT_SEED = 123                # seed de inicialização de pesos (igual em TODOS os modelos)
