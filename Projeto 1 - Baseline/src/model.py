"""
model.py
========
Define a arquitetura do MLP usando torch.nn.

Arquitetura
-----------
Um MLP totalmente conectado (fully connected) com um número configurável
de camadas ocultas (todas com ativação tanh) e uma camada de saída
LINEAR (sem ativação) - adequada para regressão, já que y pode assumir
qualquer valor real.

Por que a mesma classe serve para o baseline E para as ablações?
--------------------------------------------------------------------
O enunciado exige que "os modelos aditivados não devem sofrer alterações
em suas arquiteturas". Por isso, a classe abaixo tem SEMPRE a mesma
sequência de camadas (mesmo número de neurônios, mesma ativação). O
único "interruptor" que muda entre o baseline e a ablação de dropout é
o parâmetro `dropout_rate`:
  - dropout_rate = 0.0  -> nn.Dropout(0.0) não desliga nenhum neurônio,
    ou seja, se comporta como se a camada nem existisse (é a
    configuração do baseline e das ablações de L1/L2/momentum).
  - dropout_rate > 0.0  -> passa a zerar aleatoriamente uma fração dos
    neurônios ocultos durante o treino (ablação de dropout).

L1, L2 e momentum NÃO alteram a arquitetura (nn.Module) de forma
nenhuma - eles são implementados fora do modelo, respectivamente como
um termo extra na função de perda (L1) e como parâmetros do otimizador
torch.optim.SGD (L2 = weight_decay, momentum = momentum). Isso é
proposital: mantém a rede idêntica em todos os experimentos, mudando
apenas o processo de treinamento, exatamente como pedido no enunciado.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.0, seed=0):
        """
        layer_sizes : lista de inteiros, ex.: [1, 16, 1]
                      -> 1 entrada, 1 camada oculta com 16 neurônios, 1 saída.
                      Pode ter mais de uma camada oculta, ex.: [1, 32, 16, 1].
        dropout_rate: probabilidade de "desligar" um neurônio oculto
                      durante o treino (0.0 = dropout desligado).
        seed        : semente do gerador aleatório do PyTorch, usada na
                      inicialização dos pesos. Usamos a MESMA seed no
                      baseline e em todas as ablações, para que os
                      modelos partam do mesmo ponto inicial e a única
                      diferença entre eles seja o componente estudado.
        """
        super().__init__()
        torch.manual_seed(seed)

        layers = []
        n_hidden_layers = len(layer_sizes) - 2  # descontando entrada e saída
        for i in range(len(layer_sizes) - 1):
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
            layers.append(nn.Linear(in_f, out_f))
            is_last = (i == len(layer_sizes) - 2)
            if not is_last:
                # Camadas ocultas: ativação tanh (não-linearidade clássica
                # de MLPs "vanilla", como pedido no enunciado - nada de
                # ReLU/GELU sofisticadas para o baseline).
                layers.append(nn.Tanh())
                # nn.Dropout(p=0.0) é, na prática, a função identidade,
                # então incluí-la aqui não muda em nada o baseline; ela só
                # passa a ter efeito quando dropout_rate > 0 (ablação).
                layers.append(nn.Dropout(p=dropout_rate))
            # a camada de saída (is_last=True) fica LINEAR: sem ativação,
            # pois é regressão.

        self.net = nn.Sequential(*layers)
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate

    def forward(self, x):
        return self.net(x)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())
