"""
trainer.py
==========
Loop de treinamento genérico (mini-batch SGD, via torch.optim.SGD) usado tanto pela busca empírica do baseline quanto pelo treino 
final de todos os modelos (baseline + 4 ablações). Centralizar essa lógica em um único lugar garante que TODOS os modelos sejam 
treinados exatamente da mesma forma (mesmo número de épocas, mesmo tamanho de lote, mesma forma de embaralhar os dados a cada época)
- a única coisa que muda de um modelo para o outro são os hiperparâmetros do otimizador (momentum, weight_decay=L2) e o termo extra 
de L1 somado manualmente à perda.

Por que torch.optim.SGD (e não Adam)?
----------------------------------------
torch.optim.SGD com momentum=0 e weight_decay=0 implementa exatamente a regra clássica: w := w - lr * grad
sem nenhuma adaptação de taxa de aprendizado por parâmetro (ao contrário de Adam/RMSProp), o que é a definição de "SGD puro".

Como L1, L2 e momentum são "ligados", sem alterar a arquitetura:
--------------------------------------------------------------------
- momentum: passado como argumento momentum=... ao construtor de torch.optim.SGD. Com momentum=0 (baseline), o otimizador se comporta
  como SGD puro; com momentum>0 (ablação), o otimizador acumula uma "velocidade" com base nos gradientes anteriores.
- L2 (weight decay): passado como argumento weight_decay=... ao construtor de torch.optim.SGD. O PyTorch implementa isso somando
  weight_decay * w ao gradiente antes do passo de atualização - matematicamente idêntico a adicionar (weight_decay/2)*||w||^2 à função
  de perda.
- L1: o PyTorch NÃO tem um argumento pronto para L1 no otimizador (ao contrário do L2), então o somamos manualmente ao valor da perda
  antes de chamar .backward(): loss = MSE(y_pred, y_true) + l1_lambda * sum(|w| para cada peso) Isso faz com que o autograd do PyTorch
  calcule automaticamente o gradiente correto (subgradiente sign(w)) da penalidade L1 durante o backward, sem precisarmos derivar isso
  manualmente.
"""

import numpy as np
import torch
import torch.nn as nn


def l1_penalty(model):
    """Soma |w| de todos os PESOS (não dos vieses/bias) do modelo. Por convenção padrão, a regularização L1/L2 é aplicada apenas aos
    pesos das conexões, não aos biases (os biases não contribuem para a "complexidade" do modelo da mesma forma que os pesos)."""
    penalty = 0.0
    for name, param in model.named_parameters(): 
        if "weight" in name:
            penalty = penalty + param.abs().sum()
    return penalty


def train(model, X_train, y_train, X_val, y_val, lr, epochs, batch_size=8, momentum=0.0, weight_decay=0.0, l1_lambda=0.0,
        seed=0, verbose=False, log_every=50, restore_best=True):
    """
    Treina `model` (nn.Module) por `epochs` épocas usando mini-batch SGD.

    restore_best : se True (padrão), ao final do treino os pesos do modelo são restaurados para o estado que obteve a MENOR perda de validação durante 
        todo o treinamento (checkpointing do melhor ponto de validação, também chamado de "early stopping" na prática). Isso é aplicado da MESMA forma
        a TODOS os modelos do projeto (baseline e as 4 ablações), então não favorece nenhuma técnica em particular - é apenas uma forma padrão e neutra 
        de escolher em qual época "parar de olhar" para o modelo, evitando reportar métricas de um ponto já claramente overfitado ao final do treino.

    Por que mini-batch (batch_size=8) e não batch completo ou 1 amostra por vez?
    -------------------------------------------------------------------
    - Batch completo (as 30 amostras de treino de uma vez) faz o gradiente ser mais "suave", mas dá poucos passos de atualização por época.
    - 1 amostra por vez é bastante ruidoso com um dataset tão pequeno. 
    - mini-batch é o meio-termo padrão adotado na prática, e o que se costuma chamar de "SGD" no contexto de redes neurais.

    Retorna um dicionário 'history' com as curvas de perda (MSE "puro", sem os termos de regularização) por época, para treino e validação -
    usadas depois para os gráficos de evolução do treinamento.
    """
    torch.manual_seed(seed) 
    mse_loss_fn = nn.MSELoss() 

    #otimizador que controla como os pesos serão atualizados a cada passo de SGD. lr é a velocidade de aprendizado, weight_decay aplica L2, 
    #momentum usa informações dos gradientes anteriores para acelerar a convergência.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay) 
 
    n = X_train.shape[0] # número de amostras de treino
    history = {"train_loss": [], "val_loss": []}  

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1): #uma época = uma passagem completa por todas as amostras de treino
        model.train()  # ativa o dropout (se dropout_rate > 0)

        # Embaralha os índices de treino a cada época (prática padrão de SGD: evita que a rede "decore" a ordem fixa dos dados).
        perm = torch.randperm(n)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for start in range(0, n, batch_size): #dados divididos em pequenos grupos (mini-batches). Cada mini-batch é usado para um passo de atualização dos pesos.
            end = start + batch_size 
            X_batch = X_shuffled[start:end] 
            y_batch = y_shuffled[start:end]

            optimizer.zero_grad()                   # zera gradientes acumulados do passo anterior
            y_pred = model(X_batch)                 # forward pass: rede recebe os dados e faz a predição
            loss = mse_loss_fn(y_pred, y_batch)     #compara o que a rede previu com o valor verdadeiro, quanto menor MSE melhor a predição

            if l1_lambda > 0.0: #se L1 estiver ativado, adiciona o termo de penalidade L1 à perda antes do backward
                loss = loss + l1_lambda * l1_penalty(model)

            loss.backward()                    # backpropagation (autograd): calcula como cada peso contribuiu para o erro, e armazena o gradiente em cada tensor de peso
            optimizer.step()                   # atualização dos pesos (regra do SGD) para tentar diminuir o erro na próxima iteração

        # Ao final de cada época, medimos o desempenho em treino e validação com model.eval() (desliga o dropout) e sem calcular gradiente (torch.no_grad(), 
        # mais rápido e sem gastar memória). Importante: aqui usamos APENAS o MSE "puro" (sem L1/L2), pois queremos comparar a capacidade preditiva real dos
        # modelos, e não o valor da função-objetivo de otimização (que é inflado artificialmente pelos termos de regularização).

        model.eval()

        with torch.no_grad():
            train_loss = mse_loss_fn(model(X_train), y_train).item()    #erro nos dados de treino (MSE "puro", sem L1/L2)
            val_loss = mse_loss_fn(model(X_val), y_val).item()          #erro nos dados de validação (MSE "puro", sem L1/L2)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss: #verifica se o modelo melhorou na validação
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()} #se sim, armanzena os pesos

        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(f"  época {epoch:4d}/{epochs} | "
                  f"loss treino={train_loss:.4f} | loss val={val_loss:.4f}")

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss

    if restore_best and best_state is not None: 
        model.load_state_dict(best_state) #volta para os pesos da época onde teve o menor erro de validação

    return history #retorna as perdas de treino e validação por época, para plotar depois a evolução do treinamento
