"""
CNN x MLP EM CLASSIFICAÇÃO DE IMAGENS  (Fashion-MNIST por padrão; também MNIST e CIFAR-10)
==========================================================================================
O que o script faz:
  1) treina 8 topologias de CNN x 2 taxas de aprendizado (16 treinos) e escolhe pela VALIDAÇÃO;
  2) pega as 2 melhores topologias, calcula a MATRIZ DE CONFUSÃO de cada uma no teste e avalia a
     DIFERENÇA entre elas (teste de McNemar, intervalo de confiança da diferença e acurácia por classe);
  3) treina MLPs (4 topologias x 2 taxas) e compara com a CNN em ACURÁCIA e NÚMERO DE PARÂMETROS,
     inclusive com um MLP "casado" que tem o mesmo número de parâmetros da melhor CNN.

Uso:
    python cnn_classificacao.py                    # Fashion-MNIST, busca completa
    python cnn_classificacao.py --dataset cifar10  # ou: mnist
    python cnn_classificacao.py --rapido           # versão reduzida, para testar se tudo funciona

Dependências:
    pip install torch torchvision numpy pandas matplotlib

Saídas (pasta resultados_cnn_<dataset>/):
    busca_cnn.csv, busca_mlp.csv     -> todas as configurações (topologia x lr): parâmetros, custo e acurácia de validação
    comparacao_final.csv             -> CNN top-1, CNN top-2, MLP melhor e MLP casado: acurácia no teste e parâmetros
    diferenca_top2.csv               -> McNemar, IC da diferença de acurácia e acurácia por classe dos 2 melhores
    busca_topologias.png             -> acurácia de validação de cada topologia (CNN e MLP)
    curvas_aprendizagem.png          -> curvas de custo e acurácia (CNN top-1, CNN top-2 e MLP)
    matrizes_confusao.png            -> matriz de confusão no teste dos 2 melhores modelos
    acuracia_por_classe.png          -> acurácia por classe: CNN top-1, CNN top-2 e MLP
    acuracia_vs_parametros.png       -> acurácia x nº de parâmetros (CNN e MLP)
    erros_exemplos.png               -> exemplos de imagens que a melhor CNN errou
"""
import os
import time
import copy
import math
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIGURAÇÃO DO EXPERIMENTO
# ---------------------------------------------------------------------
SEED = 42
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "fashion": dict(classe="FashionMNIST", nomes=["Camiseta/top", "Calça", "Pulôver", "Vestido", "Casaco",
                                                  "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]),
    "mnist": dict(classe="MNIST", nomes=[str(i) for i in range(10)]),
    "cifar10": dict(classe="CIFAR10", nomes=["Avião", "Automóvel", "Pássaro", "Gato", "Cervo",
                                             "Cachorro", "Sapo", "Cavalo", "Navio", "Caminhão"]),
}

# Topologias de CNN: blocos = [(nº de convoluções, filtros, tamanho do kernel), ...]; cada bloco termina em MaxPool 2x2.
TOPOLOGIAS_CNN = {
    "A: 1 conv (32)":              dict(blocos=[(1, 32, 3)], denso=128, bn=False, drop=0.0),
    "B: 2 conv (32-64)":           dict(blocos=[(1, 32, 3), (1, 64, 3)], denso=128, bn=False, drop=0.0),
    "C: 2 conv (32-64) k=5":       dict(blocos=[(1, 32, 5), (1, 64, 5)], denso=128, bn=False, drop=0.0),
    "D: 2 conv (64-128)":          dict(blocos=[(1, 64, 3), (1, 128, 3)], denso=256, bn=False, drop=0.0),
    "E: 3 conv (32-64-128)":       dict(blocos=[(1, 32, 3), (1, 64, 3), (1, 128, 3)], denso=128, bn=False, drop=0.0),
    "F: 2 conv (32-64) + BN":      dict(blocos=[(1, 32, 3), (1, 64, 3)], denso=128, bn=True, drop=0.0),
    "G: 2 conv (32-64) + BN + DO": dict(blocos=[(1, 32, 3), (1, 64, 3)], denso=128, bn=True, drop=0.3),
    "H: 2x2 conv (32-64) + BN + DO": dict(blocos=[(2, 32, 3), (2, 64, 3)], denso=256, bn=True, drop=0.4),
}
# Topologias de MLP: neurônios das camadas ocultas
TOPOLOGIAS_MLP = {
    "M1: 256": [256],
    "M2: 512-256": [512, 256],
    "M3: 1024-512-256": [1024, 512, 256],
    "M4: 2048-1024-512": [2048, 1024, 512],
}
TAXAS = [1e-3, 3e-3]  # learning rate do Adam (parâmetro ajustado, além da topologia)
BATCH = 128
MAX_EPOCAS = 30
PACIENCIA = 5  # early stopping
FRACAO_VALIDACAO = 0.2
N_SEMENTES = 3  # os modelos finais são treinados com várias sementes


# =====================================================================
# 1) DADOS
# =====================================================================
def carregar_dados(nome):
    from torchvision import datasets
    classe = getattr(datasets, DATASETS[nome]["classe"])
    tr, te = classe("dados", train=True, download=True), classe("dados", train=False, download=True)

    def tensores(ds):
        X = torch.as_tensor(np.asarray(ds.data))
        X = X.unsqueeze(1) if X.ndim == 3 else X.permute(0, 3, 1, 2)  # -> (N, canais, altura, largura)
        return X.float().div(255), torch.as_tensor(np.array(ds.targets)).long()

    X, y = tensores(tr)
    Xte, yte = tensores(te)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(X), generator=g)
    n_val = int(FRACAO_VALIDACAO * len(X))
    iv, it = perm[:n_val], perm[n_val:]
    mu = X[it].mean(dim=(0, 2, 3), keepdim=True)  # padronização por canal, só com estatísticas do TREINO
    sd = X[it].std(dim=(0, 2, 3), keepdim=True)
    norm = lambda a: ((a - mu) / sd).to(DISPOSITIVO)
    return dict(tr=dict(X=norm(X[it]), y=y[it].to(DISPOSITIVO)), va=dict(X=norm(X[iv]), y=y[iv].to(DISPOSITIVO)),
                te=dict(X=norm(Xte), y=yte.to(DISPOSITIVO)), te_bruto=Xte, entrada=tuple(X.shape[1:]))


# =====================================================================
# 2) MODELOS: CNN e MLP
# =====================================================================
class CNN(nn.Module):
    def __init__(self, entrada, n_classes, blocos, denso, bn, drop):
        super().__init__()
        c, h, w = entrada
        camadas = []
        for n_conv, filtros, k in blocos:
            for _ in range(n_conv):
                camadas.append(nn.Conv2d(c, filtros, k, padding=k // 2))  # padding "same"
                if bn:
                    camadas.append(nn.BatchNorm2d(filtros))
                camadas.append(nn.ReLU())
                c = filtros
            camadas.append(nn.MaxPool2d(2))  # reduz a imagem à metade
            h, w = h // 2, w // 2
        self.conv = nn.Sequential(*camadas)
        self.classificador = nn.Sequential(nn.Flatten(), nn.Linear(c * h * w, denso), nn.ReLU(),
                                           nn.Dropout(drop), nn.Linear(denso, n_classes))

    def forward(self, x):
        return self.classificador(self.conv(x))


class MLP(nn.Module):
    def __init__(self, entrada, n_classes, ocultas):
        super().__init__()
        d = int(np.prod(entrada))
        camadas = [nn.Flatten()]
        for h in ocultas:
            camadas += [nn.Linear(d, h), nn.ReLU()]
            d = h
        camadas.append(nn.Linear(d, n_classes))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)


def milhar(n):
    return f"{n:,d}".replace(",", ".")  # 1234567 -> 1.234.567


def n_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


def largura_mlp_casado(params_alvo, entrada, n_classes):
    """Largura h de um MLP com 2 camadas ocultas [h, h] que tem ~params_alvo parâmetros."""
    d, c = int(np.prod(entrada)), n_classes
    b = d + c + 2  # params = h^2 + b*h + c
    return max(8, int(round((-b + math.sqrt(b * b - 4 * (c - params_alvo))) / 2)))


@torch.no_grad()
def logits_de(modelo, X, bloco=2000):
    modelo.eval()
    return torch.cat([modelo(X[i:i + bloco]) for i in range(0, len(X), bloco)])


def custo_acc(modelo, X, y):
    lg = logits_de(modelo, X)
    return F.cross_entropy(lg, y).item(), (lg.argmax(1) == y).float().mean().item()


# =====================================================================
# 3) TREINO COM EARLY STOPPING
# =====================================================================
def treinar(construir, lr, d, seed, max_epocas, paciencia):
    torch.manual_seed(seed)
    modelo = construir().to(DISPOSITIVO)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    Xtr, ytr = d["tr"]["X"], d["tr"]["y"]
    n, hist, melhor, estado, ep_melhor, sem_melhora = len(ytr), [], float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(1, max_epocas + 1):
        modelo.train()
        perm = torch.randperm(n, device=DISPOSITIVO)
        soma, acertos = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:  # a BatchNorm não aceita lote de 1 imagem
                continue
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            lg = modelo(xb)
            loss = F.cross_entropy(lg, yb)
            loss.backward()
            opt.step()
            soma += loss.item() * len(idx)
            acertos += (lg.argmax(1) == yb).sum().item()
        v_loss, v_acc = custo_acc(modelo, d["va"]["X"], d["va"]["y"])
        hist.append(dict(epoca=ep, loss_treino=soma / n, acc_treino=acertos / n, loss_val=v_loss, acc_val=v_acc))
        if v_loss < melhor - 1e-4:
            melhor, ep_melhor, sem_melhora, acc_m = v_loss, ep, 0, v_acc
            estado = copy.deepcopy(modelo.state_dict())
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                break
    modelo.load_state_dict(estado)  # volta para a melhor época
    return dict(modelo=modelo, hist=hist, val_loss=melhor, val_acc=acc_m, ep_melhor=ep_melhor, ep_rodadas=len(hist),
                tempo=time.time() - t0, params=n_parametros(modelo), lr=lr)


# =====================================================================
# 4) MÉTRICAS E COMPARAÇÃO ENTRE MODELOS
# =====================================================================
def matriz_confusao(y, pred, n):
    m = np.zeros((n, n), dtype=int)
    np.add.at(m, (y, pred), 1)  # linha = classe real, coluna = classe prevista
    return m


def metricas(y, pred, n):
    cm = matriz_confusao(y, pred, n)
    tp = np.diag(cm).astype(float)
    prec, rev = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rev / np.maximum(prec + rev, 1e-12)
    return dict(acuracia=tp.sum() / cm.sum(), f1_macro=f1.mean()), rev, cm  # rev = acurácia por classe


def mcnemar_exato(ok_a, ok_b):
    """Teste de McNemar (exato, bicaudal) para dois classificadores avaliados nas MESMAS imagens.
    b = só A acerta; c = só B acerta. Sob H0 (mesma taxa de erro), b ~ Binomial(b+c, 1/2)."""
    b, c = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def ic_diferenca(ok_a, ok_b, n_boot=2000):
    """IC de 95% (bootstrap pareado) da diferença de acurácia A - B."""
    rng = np.random.default_rng(SEED)
    n = len(ok_a)
    dif = [ok_a[i].mean() - ok_b[i].mean() for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return np.percentile(dif, [2.5, 97.5])


# =====================================================================
# 5) FIGURAS
# =====================================================================
def figuras(PASTA, nomes, d, df_cnn, df_mlp, finais, cnn_top, mlp_nome, erros):
    n_cls = len(nomes)
    # --- busca por topologia ---
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for a, df, titulo in [(ax[0], df_cnn, "CNN"), (ax[1], df_mlp, "MLP")]:
        piv = df.pivot_table(index="topologia", columns="lr", values="val_acc", sort=False)
        params = df.groupby("topologia", sort=False).params.first()
        x = np.arange(len(piv))
        for k, lr in enumerate(piv.columns):
            a.bar(x + (k - 0.5) * 0.38, piv[lr], 0.38, label=f"lr = {lr:g}")
        a.set_xticks(x)
        a.set_xticklabels([f"{t}\n{params[t] / 1e3:,.0f} mil par.".replace(",", ".") for t in piv.index], fontsize=7, rotation=20, ha="right")
        a.set_ylim(max(0, piv.values.min() - 0.03), min(1.0, piv.values.max() + 0.015))
        a.set_ylabel("acurácia de validação"); a.set_title(f"Topologias de {titulo}"); a.legend()
    plt.tight_layout(); plt.savefig(f"{PASTA}/busca_topologias.png", dpi=110); plt.close()

    # --- curvas de aprendizagem ---
    modelos = [(f"CNN top-1: {finais['cnn1']['nome']}", finais["cnn1"]), (f"CNN top-2: {finais['cnn2']['nome']}", finais["cnn2"]),
               (f"MLP: {mlp_nome}", finais["mlp"])]
    fig, ax = plt.subplots(2, 3, figsize=(16, 7))
    for j, (titulo, f) in enumerate(modelos):
        h = pd.DataFrame(f["hist"])
        ax[0, j].plot(h.epoca, h.loss_treino, label="treino"); ax[0, j].plot(h.epoca, h.loss_val, label="validação")
        ax[0, j].axvline(f["ep_melhor"], ls="--", color="gray", label="melhor época")
        ax[0, j].set_title(titulo, fontsize=9); ax[0, j].set_ylabel("entropia cruzada"); ax[0, j].legend()
        ax[1, j].plot(h.epoca, h.acc_treino, label="treino"); ax[1, j].plot(h.epoca, h.acc_val, label="validação")
        ax[1, j].axvline(f["ep_melhor"], ls="--", color="gray"); ax[1, j].set_xlabel("época"); ax[1, j].set_ylabel("acurácia")
        ax[1, j].legend()
    plt.tight_layout(); plt.savefig(f"{PASTA}/curvas_aprendizagem.png", dpi=110); plt.close()

    # --- matrizes de confusão dos 2 melhores ---
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.2))
    for a, chave in zip(ax, ["cnn1", "cnn2"]):
        f = finais[chave]
        cm = f["cm"]; pct = cm / cm.sum(1, keepdims=True)
        im = a.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(n_cls):
            for j in range(n_cls):
                if cm[i, j] > 0:
                    a.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7,
                           color="white" if pct[i, j] > 0.5 else "black")
        a.set_xticks(range(n_cls)); a.set_xticklabels(nomes, rotation=40, ha="right", fontsize=8)
        a.set_yticks(range(n_cls)); a.set_yticklabels(nomes, fontsize=8)
        a.set_xlabel("classe prevista"); a.set_ylabel("classe real")
        a.set_title(f"{'CNN top-1' if chave == 'cnn1' else 'CNN top-2'}: {f['nome']}\nacurácia no teste = {f['acc_seed0']:.4f}", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{PASTA}/matrizes_confusao.png", dpi=110); plt.close()

    # --- acurácia por classe ---
    fig, ax = plt.subplots(figsize=(12, 4.2))
    w = 0.27
    for k, (rot, chave) in enumerate([("CNN top-1", "cnn1"), ("CNN top-2", "cnn2"), ("MLP", "mlp")]):
        ax.bar(np.arange(n_cls) + (k - 1) * w, finais[chave]["rev"], w, label=rot)
    ax.set_xticks(range(n_cls)); ax.set_xticklabels(nomes, rotation=25, ha="right")
    ax.set_ylabel("acurácia no teste"); ax.set_ylim(max(0, min(finais[c]["rev"].min() for c in ["cnn1", "cnn2", "mlp"]) - 0.05), 1.01)
    ax.legend(); ax.set_title("Acurácia por classe")
    plt.tight_layout(); plt.savefig(f"{PASTA}/acuracia_por_classe.png", dpi=110); plt.close()

    # --- acurácia x parâmetros ---
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(df_cnn.params, df_cnn.val_acc, color="tab:blue", label="CNN (cada topologia x lr)", s=40)
    ax.scatter(df_mlp.params, df_mlp.val_acc, color="tab:orange", label="MLP (cada topologia x lr)", s=40)
    ax.scatter(finais["mlp_casado"]["params"], finais["mlp_casado"]["val_acc"], color="tab:green", marker="D", s=70,
               label="MLP com nº de parâmetros da CNN top-1")
    ax.set_xscale("log"); ax.set_xlabel("número de parâmetros (escala log)"); ax.set_ylabel("acurácia de validação")
    ax.set_title("Acurácia x número de parâmetros"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{PASTA}/acuracia_vs_parametros.png", dpi=110); plt.close()

    # --- exemplos de erros da melhor CNN ---
    idx, pred = erros["idx"][:16], erros["pred"][:16]
    y_te, Xb = d["te"]["y"].cpu().numpy(), d["te_bruto"]
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.6))
    for a, i, p in zip(ax.ravel(), idx, pred):
        im = Xb[i].numpy()
        a.imshow(im[0] if im.shape[0] == 1 else im.transpose(1, 2, 0), cmap="gray" if im.shape[0] == 1 else None)
        a.set_title(f"real: {nomes[y_te[i]]}\nprev.: {nomes[p]}", fontsize=7); a.axis("off")
    plt.suptitle(f"Exemplos de erros da melhor CNN ({finais['cnn1']['nome']})")
    plt.tight_layout(); plt.savefig(f"{PASTA}/erros_exemplos.png", dpi=110); plt.close()


# =====================================================================
# 6) PROGRAMA PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="fashion")
    ap.add_argument("--rapido", action="store_true", help="busca reduzida (teste rápido)")
    ap.add_argument("--epocas", type=int, default=MAX_EPOCAS)
    ap.add_argument("--sementes", type=int, default=N_SEMENTES)
    args = ap.parse_args()

    nomes = DATASETS[args.dataset]["nomes"]
    PASTA = f"resultados_cnn_{args.dataset}"
    top_cnn, top_mlp, taxas, max_ep, n_sem = dict(TOPOLOGIAS_CNN), dict(TOPOLOGIAS_MLP), TAXAS, args.epocas, args.sementes
    if args.rapido:
        top_cnn, top_mlp = dict(list(TOPOLOGIAS_CNN.items())[:3]), dict(list(TOPOLOGIAS_MLP.items())[:2])
        taxas, max_ep, n_sem = TAXAS[:1], min(max_ep, 3), 2
    os.makedirs(PASTA, exist_ok=True)
    print(f"Dispositivo: {DISPOSITIVO} | dataset: {args.dataset}")

    d = carregar_dados(args.dataset)
    n_cls = len(nomes)
    print(f"Imagens {d['entrada']} | treino {len(d['tr']['y'])} | validação {len(d['va']['y'])} | teste {len(d['te']['y'])}")
    y_te = d["te"]["y"].cpu().numpy()

    # ---- busca de CNNs (topologia x lr), escolha pela VALIDAÇÃO ----
    linhas_cnn, res_cnn = [], {}
    print(f"\n=== CNN: {len(top_cnn) * len(taxas)} configurações ===")
    for (nome, spec), lr in itertools.product(top_cnn.items(), taxas):
        r = treinar(lambda s=spec: CNN(d["entrada"], n_cls, **s), lr, d, SEED, max_ep, PACIENCIA)
        res_cnn[(nome, lr)] = r
        linhas_cnn.append(dict(topologia=nome, lr=lr, params=r["params"], val_loss=r["val_loss"], val_acc=r["val_acc"],
                               epoca_melhor=r["ep_melhor"], epocas_rodadas=r["ep_rodadas"], tempo_s=r["tempo"]))
        print(f"  {nome:30s} lr={lr:<6g} params={milhar(r['params']):>9s}  val_acc={r['val_acc']:.4f} val_loss={r['val_loss']:.4f} "
              f"(época {r['ep_melhor']}/{r['ep_rodadas']}, {r['tempo']:.0f}s)")
    df_cnn = pd.DataFrame(linhas_cnn)
    df_cnn.to_csv(f"{PASTA}/busca_cnn.csv", index=False)

    # ---- busca de MLPs ----
    linhas_mlp, res_mlp = [], {}
    print(f"\n=== MLP: {len(top_mlp) * len(taxas)} configurações ===")
    for (nome, ocultas), lr in itertools.product(top_mlp.items(), taxas):
        r = treinar(lambda o=ocultas: MLP(d["entrada"], n_cls, o), lr, d, SEED, max_ep, PACIENCIA)
        res_mlp[(nome, lr)] = r
        linhas_mlp.append(dict(topologia=nome, lr=lr, params=r["params"], val_loss=r["val_loss"], val_acc=r["val_acc"],
                               epoca_melhor=r["ep_melhor"], epocas_rodadas=r["ep_rodadas"], tempo_s=r["tempo"]))
        print(f"  {nome:30s} lr={lr:<6g} params={milhar(r['params']):>9s}  val_acc={r['val_acc']:.4f} val_loss={r['val_loss']:.4f} "
              f"(época {r['ep_melhor']}/{r['ep_rodadas']}, {r['tempo']:.0f}s)")
    df_mlp = pd.DataFrame(linhas_mlp)
    df_mlp.to_csv(f"{PASTA}/busca_mlp.csv", index=False)

    # ---- os 2 melhores CNN (topologias DISTINTAS; melhor lr de cada uma) e o melhor MLP ----
    ordenar = lambda df: df.sort_values(["val_acc", "val_loss"], ascending=[False, True])
    melhor_por_topologia = ordenar(df_cnn).groupby("topologia", sort=False).head(1)
    top2 = ordenar(melhor_por_topologia).head(2)
    b_mlp = ordenar(df_mlp).iloc[0]
    print(f"\nTop-2 CNN: {top2.topologia.tolist()} | melhor MLP: {b_mlp.topologia}")

    def modelo_final(construir, lr, r_busca, nome, semente_extra):
        """Semente 0 reaproveita o treino da busca; as demais retreinam. Avalia no TESTE."""
        accs, f1s, r0 = [], [], None
        for s in range(n_sem):
            r = r_busca if (r_busca is not None and s == 0) else treinar(construir, lr, d, SEED + s, max_ep, PACIENCIA)
            pred = logits_de(r["modelo"], d["te"]["X"]).argmax(1).cpu().numpy()
            m, rev, cm = metricas(y_te, pred, n_cls)
            accs.append(m["acuracia"]); f1s.append(m["f1_macro"])
            if s == 0:
                r0 = dict(r, pred=pred, rev=rev, cm=cm, acc_seed0=m["acuracia"])
        return dict(r0, nome=nome, acc_media=np.mean(accs), acc_dp=np.std(accs, ddof=1) if n_sem > 1 else 0.0,
                    f1_media=np.mean(f1s), accs=accs)

    finais = {}
    for chave, (_, linha) in zip(["cnn1", "cnn2"], top2.iterrows()):
        spec = top_cnn[linha.topologia]
        print(f"\n=== {chave.upper()}: {linha.topologia} lr={linha.lr:g} ({n_sem} semente(s)) ===")
        finais[chave] = modelo_final(lambda s=spec: CNN(d["entrada"], n_cls, **s), linha.lr, res_cnn[(linha.topologia, linha.lr)], linha.topologia, 0)
        print(f"  acurácia no teste: {finais[chave]['acc_media']:.4f} ± {finais[chave]['acc_dp']:.4f}  {np.round(finais[chave]['accs'], 4)}")
    print(f"\n=== MLP melhor: {b_mlp.topologia} lr={b_mlp.lr:g} ===")
    finais["mlp"] = modelo_final(lambda o=top_mlp[b_mlp.topologia]: MLP(d["entrada"], n_cls, o), b_mlp.lr,
                                 res_mlp[(b_mlp.topologia, b_mlp.lr)], b_mlp.topologia, 0)
    print(f"  acurácia no teste: {finais['mlp']['acc_media']:.4f} ± {finais['mlp']['acc_dp']:.4f}")
    h = largura_mlp_casado(finais["cnn1"]["params"], d["entrada"], n_cls)
    print(f"\n=== MLP casado (2 camadas de {h} neurônios, ~{milhar(finais['cnn1']['params'])} parâmetros) ===")
    finais["mlp_casado"] = modelo_final(lambda: MLP(d["entrada"], n_cls, [h, h]), b_mlp.lr, None, f"{h}-{h}", 0)
    print(f"  acurácia no teste: {finais['mlp_casado']['acc_media']:.4f} ± {finais['mlp_casado']['acc_dp']:.4f}")

    # ---- comparação final: acurácia e número de parâmetros ----
    comp = pd.DataFrame([
        dict(modelo="CNN top-1", topologia=finais["cnn1"]["nome"], params=finais["cnn1"]["params"], acc_val=finais["cnn1"]["val_acc"],
             acc_teste_media=finais["cnn1"]["acc_media"], acc_teste_dp=finais["cnn1"]["acc_dp"], f1_macro=finais["cnn1"]["f1_media"], tempo_s=finais["cnn1"]["tempo"]),
        dict(modelo="CNN top-2", topologia=finais["cnn2"]["nome"], params=finais["cnn2"]["params"], acc_val=finais["cnn2"]["val_acc"],
             acc_teste_media=finais["cnn2"]["acc_media"], acc_teste_dp=finais["cnn2"]["acc_dp"], f1_macro=finais["cnn2"]["f1_media"], tempo_s=finais["cnn2"]["tempo"]),
        dict(modelo="MLP (melhor)", topologia=finais["mlp"]["nome"], params=finais["mlp"]["params"], acc_val=finais["mlp"]["val_acc"],
             acc_teste_media=finais["mlp"]["acc_media"], acc_teste_dp=finais["mlp"]["acc_dp"], f1_macro=finais["mlp"]["f1_media"], tempo_s=finais["mlp"]["tempo"]),
        dict(modelo="MLP (mesmo nº de parâmetros da CNN top-1)", topologia=finais["mlp_casado"]["nome"], params=finais["mlp_casado"]["params"],
             acc_val=finais["mlp_casado"]["val_acc"], acc_teste_media=finais["mlp_casado"]["acc_media"], acc_teste_dp=finais["mlp_casado"]["acc_dp"],
             f1_macro=finais["mlp_casado"]["f1_media"], tempo_s=finais["mlp_casado"]["tempo"])])
    comp.to_csv(f"{PASTA}/comparacao_final.csv", index=False)

    # ---- diferença entre os 2 melhores (mesmas imagens de teste, semente da busca) ----
    ok1, ok2 = finais["cnn1"]["pred"] == y_te, finais["cnn2"]["pred"] == y_te
    b, c, p = mcnemar_exato(ok1, ok2)
    lo, hi = ic_diferenca(ok1, ok2)
    dif = pd.DataFrame(dict(classe=nomes, acc_top1=finais["cnn1"]["rev"], acc_top2=finais["cnn2"]["rev"]))
    dif["diferenca"] = dif.acc_top1 - dif.acc_top2
    dif.to_csv(f"{PASTA}/diferenca_top2.csv", index=False)
    resumo_dif = dict(acc_top1=ok1.mean(), acc_top2=ok2.mean(), dif=ok1.mean() - ok2.mean(), ic_lo=lo, ic_hi=hi,
                      so_top1_acerta=b, so_top2_acerta=c, p_mcnemar=p)
    pd.DataFrame([resumo_dif]).to_csv(f"{PASTA}/mcnemar_top2.csv", index=False)

    erros_idx = np.flatnonzero(finais["cnn1"]["pred"] != y_te)
    figuras(PASTA, nomes, d, df_cnn, df_mlp, finais, top2, b_mlp.topologia,
            dict(idx=erros_idx, pred=finais["cnn1"]["pred"][erros_idx]))

    # ---- resumo no terminal ----
    print("\n===== COMPARAÇÃO FINAL (teste; média ± dp entre sementes) =====")
    print(comp.round(4).to_string(index=False))
    print("\n===== DIFERENÇA ENTRE OS 2 MELHORES (semente da busca, mesmas imagens de teste) =====")
    print(f"  acurácia top-1 = {ok1.mean():.4f} | top-2 = {ok2.mean():.4f} | diferença = {ok1.mean() - ok2.mean():+.4f}  "
          f"(IC 95%: {lo:+.4f} a {hi:+.4f})")
    print(f"  só o top-1 acerta: {b} | só o top-2 acerta: {c} | McNemar exato: p = {p:.4f}")
    print(f"  {'-> diferença estatisticamente significativa (p < 0,05)' if p < 0.05 else '-> diferença NÃO significativa (p >= 0,05)'}")
    print("\nAcurácia por classe (top-1 x top-2):")
    print(dif.round(3).to_string(index=False))
    print(f"\nArquivos salvos em: {os.path.abspath(PASTA)}")