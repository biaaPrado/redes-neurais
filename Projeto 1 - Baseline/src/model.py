"""
model.py
========
Define a arquitetura do MLP usando torch.nn.

Arquitetura
-----------
Um MLP totalmente conectado (fully connected) com um número configurável de camadas ocultas (todas com ativação tanh) e uma 
camada de saída LINEAR (sem ativação) - adequada para regressão, já que y pode assumir qualquer valor real.

Por que a mesma classe serve para o baseline E para as ablações?
--------------------------------------------------------------------
O enunciado exige que "os modelos aditivados não devem sofrer alterações em suas arquiteturas". Por isso, a classe abaixo tem
SEMPRE a mesma sequência de camadas (mesmo número de neurônios, mesma ativação). O único "interruptor" que muda entre o baseline
e a ablação de dropout é o parâmetro `dropout_rate`:
  - dropout_rate = 0.0  -> nn.Dropout(0.0) não desliga nenhum neurônio, ou seja, se comporta como se a camada nem existisse (é a
    configuração do baseline e das ablações de L1/L2/momentum).
  - dropout_rate > 0.0  -> passa a zerar aleatoriamente uma fração dos neurônios ocultos durante o treino (ablação de dropout).

L1, L2 e momentum NÃO alteram a arquitetura (nn.Module) de forma nenhuma - eles são implementados fora do modelo, respectivamente 
como um termo extra na função de perda (L1) e como parâmetros do otimizador torch.optim.SGD (L2 = weight_decay, momentum = momentum). 
Isso é proposital: mantém a rede idêntica em todos os experimentos, mudando apenas o processo de treinamento.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.0, seed=0):
        """
        layer_sizes : lista de inteiros, ex.: [1, 16, 1]
                      -> 1 entrada, 1 camada oculta com 16 neurônios, 1 saída. Pode ter mais de uma camada oculta, ex.: [1, 32, 16, 1].
        dropout_rate: probabilidade de "desligar" um neurônio oculto durante o treino (0.0 = dropout desligado).
        seed        : semente do gerador aleatório do PyTorch, usada na inicialização dos pesos. Usamos a MESMA seed no baseline e 
                      em todas as ablações, para que os modelos partam do mesmo ponto inicial e a única diferença entre eles seja o 
                      componente estudado.
        """
        super().__init__() 
        torch.manual_seed(seed)  # garante que a inicialização dos pesos seja a mesma em todas as execuções (baseline e ablações)

        layers = []
        n_hidden_layers = len(layer_sizes) - 2  # descontando entrada e saída
        for i in range(len(layer_sizes) - 1):    # cria uma camada para cada par de tamanhos consecutivos (entrada->oculta, oculta->oculta, oculta->saída)
            in_f, out_f = layer_sizes[i], layer_sizes[i + 1]  
            layers.append(nn.Linear(in_f, out_f)) #conexão das camadas e multiplicação por pesos (Linear = fully connected)
            is_last = (i == len(layer_sizes) - 2) #encontrar a última camada (a de saída) para não colocar ativação nela, pois é regressão e queremos que a saída seja linear.
            if not is_last: #enquanto não for a última camada, adiciona ativação tanh e dropout (se dropout_rate > 0)
                layers.append(nn.Tanh()) #camadas ocultas: ativação tanh (não-linearidade clássica de MLPs "vanilla")
                layers.append(nn.Dropout(p=dropout_rate)) #só adiciona dropout se dropout_rate > 0.0, caso contrário não desliga nenhum neurônio e se comporta como se a camada nem existisse.
            #quando encontra a ultima camada, não adiciona ativação nem dropout, pois queremos que a saída seja linear (regressão).

        self.net = nn.Sequential(*layers)  #lista das camadas, que serão executadas sequencialmente no forward pass
        self.layer_sizes = layer_sizes     
        self.dropout_rate = dropout_rate

    def forward(self, x): #define que para cada x recebido, a saída será o resultado da rede (self.net) aplicada a x.
        return self.net(x)  

    def n_parameters(self): #conta o número total de parâmetros treináveis na rede (pesos e bias de todas as camadas)
        return sum(p.numel() for p in self.parameters())
