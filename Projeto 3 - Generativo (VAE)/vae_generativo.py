"""
VAE (Variational Autoencoder) GENERATIVO
=========================================
Uso:
    python vae_generativo.py              # busca completa (pode demorar; use GPU se tiver)
    python vae_generativo.py --rapido     # versão reduzida, para testar se tudo funciona
    python vae_generativo.py --refinar 150 --paciencia 10
        # SEM refazer a busca: lê os 2 melhores modelos de resultados_vae/ e os retreina
        # com até 150 épocas. Os resultados novos vão para resultados_vae_refinado/
 
Dependências:
    pip install torch torchvision numpy pandas matplotlib scikit-learn
 
Saídas (pasta resultados_vae/):
    busca_hiperparametros.csv   -> todas as configurações testadas (latente x topologia x lr x batch)
    resumo_final.csv            -> melhor modelo de cada latente: custo (val e teste)
    metricas_latente.csv        -> clusters / separação de rótulos no espaço latente
    variancia_pca_8d.csv        -> variância explicada por cada componente principal (8D)
    busca_hiperparametros.png   -> custo de validação de cada configuração
    curvas_treino.png           -> curvas de treino/validação dos melhores modelos
    latente_2d.png              -> espaço latente de 2 dimensões, colorido por rótulo
    latente_8d_pca.png          -> projeção PCA do espaço latente de 8D + variância explicada
    reconstrucoes.png           -> original x reconstrução (2D e 8D)
    geracao.png                 -> imagens NOVAS geradas amostrando z ~ N(0, I)
    variedade_2d.png            -> imagens geradas varrendo uma grade do espaço latente 2D
"""

import os
import sys
import time
import copy
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier

#Configurações gerais
SEED = 42
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASTA = "resultados_vae"  #pasta de saída (no modo --refinar vira resultados_vae_refinado)
PASTA_BUSCA = "resultados_vae"  #onde está a busca já feita (usada pelo --refinar)
NOMES = ["Camiseta/top", "Calça", "Pulôver", "Vestido", "Casaco", "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]
 
LATENTES = [2, 8]  #duas dimensões do espaço latente (2D e 8D)
BETA = 1.0  # peso do KL (1.0 = VAE padrão)
 
# Espaço de busca de hiperparâmetros (a topologia pode mudar para cada latente)
ARQUITETURAS = {  # nome -> neurônios das camadas ocultas do encoder (o decoder é o espelho)
    "256": [256],
    "256-128": [256, 128],
    "512-256": [512, 256],
    "512-256-128": [512, 256, 128],
}
TAXAS = [1e-3, 5e-4]  # learning rate do Adam
BATCHES = [128, 256]
FRACAO_VALIDACAO = 0.2  # 20% dos 60.000 de treino viram validação

#1 - Carregamento de dados
def carregar_dados(sintetico=False):
    """Retorna X, y (treino+validação) e X_teste, y_teste. Pixels em [0,1], achatados (784)."""
    if sintetico:  # só para testar o código sem baixar nada
        g = torch.Generator().manual_seed(SEED)
        proto = torch.rand(10, 784, generator=g) ** 2
 
        def gerar(n):
            y = torch.randint(0, 10, (n,), generator=g)
            return (proto[y] + 0.15 * torch.randn(n, 784, generator=g)).clamp(0, 1), y
 
        X, y = gerar(12000)
        Xt, yt = gerar(2000)
        return X, y, Xt, yt
    from torchvision import datasets
    tr = datasets.FashionMNIST("dados", train=True, download=True)
    te = datasets.FashionMNIST("dados", train=False, download=True)
    f = lambda d: d.data.float().div(255).view(-1, 784)  # normaliza para [0, 1]
    return f(tr), tr.targets, f(te), te.targets
 
 
def dividir(X, y):
    """Separa treino e validação (aleatório, com semente fixa)."""
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(X), generator=g)
    n_val = int(FRACAO_VALIDACAO * len(X))
    v, t = perm[:n_val], perm[n_val:]
    return X[t], y[t], X[v], y[v]


#2 - Modelo VAE
class VAE(nn.Module):
    def __init__(self, latente, ocultas):
        super().__init__()
        camadas, entrada = [], 784      #encoder: 784 -> camadas ocultas -> (mu, logvar)
        for h in ocultas:
            camadas += [nn.Linear(entrada, h), nn.ReLU()]
            entrada = h
        self.enc = nn.Sequential(*camadas)
        self.mu = nn.Linear(entrada, latente)  #média da gaussiana q(z|x)
        self.logvar = nn.Linear(entrada, latente)  #log da variância de q(z|x)
        
        camadas, entrada = [], latente  #decoder: z -> camadas ocultas (espelho) -> 784 logits
        for h in reversed(ocultas):
            camadas += [nn.Linear(entrada, h), nn.ReLU()]
            entrada = h
        camadas.append(nn.Linear(entrada, 784))  #sem sigmoid: a sigmoid está na função de custo
        self.dec = nn.Sequential(*camadas)
 
    def codificar(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)
 
    def reparametrizar(self, mu, logvar):
        """z = mu + sigma * epsilon, com epsilon ~ N(0,I): permite derivar através da amostragem."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
 
    def forward(self, x):
        mu, logvar = self.codificar(x)
        z = self.reparametrizar(mu, logvar)
        return self.dec(z), mu, logvar
 
 
def custo(logits, x, mu, logvar, beta=BETA):
    """Custo do VAE (negativo do ELBO) = reconstrução + beta * KL. Soma sobre o lote."""
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL( q(z|x) || N(0,I) )
    return rec + beta * kl, rec, kl
 
 
@torch.no_grad()
def avaliar(modelo, X, bloco=5000):
    """Custo médio POR AMOSTRA (total, reconstrução, KL) em um conjunto."""
    modelo.eval()
    tot = rec = kl = 0.0
    for i in range(0, len(X), bloco):
        xb = X[i:i + bloco]
        logits, mu, lv = modelo(xb)
        l, r, k = custo(logits, xb, mu, lv)
        tot, rec, kl = tot + l.item(), rec + r.item(), kl + k.item()
    n = len(X)
    return tot / n, rec / n, kl / n


#3 - Treinamento com Early Stopping + Busca de Hiperparâmetros
def treinar(latente, ocultas, lr, batch, Xtr, Xval, max_epocas, paciencia):
    torch.manual_seed(SEED)
    modelo = VAE(latente, ocultas).to(DISPOSITIVO)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    n = len(Xtr)
    hist, melhor, estado, ep_melhor, sem_melhora = [], float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(1, max_epocas + 1):
        modelo.train()
        perm = torch.randperm(n, device=DISPOSITIVO)
        soma = 0.0
        for i in range(0, n, batch):
            xb = Xtr[perm[i:i + batch]]
            opt.zero_grad()
            logits, mu, lv = modelo(xb)
            loss, _, _ = custo(logits, xb, mu, lv)
            (loss / len(xb)).backward()  # gradiente do custo médio por amostra
            opt.step()
            soma += loss.item()
        val, rec, kl = avaliar(modelo, Xval)
        hist.append(dict(epoca=ep, treino=soma / n, validacao=val))
        if val < melhor - 1e-3:  # melhorou: guarda este modelo
            melhor, rec_m, kl_m = val, rec, kl
            estado, ep_melhor, sem_melhora = copy.deepcopy(modelo.state_dict()), ep, 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:  # early stopping
                break
    modelo.load_state_dict(estado)  # volta para a MELHOR época (menor custo de validação)
    return dict(modelo=modelo, hist=hist, val=melhor, val_rec=rec_m, val_kl=kl_m, ep_melhor=ep_melhor, ep_rodadas=len(hist), tempo=time.time() - t0)
 
 
def buscar(latente, Xtr, Xval, arquiteturas, taxas, batches, max_epocas, paciencia):
    linhas, melhor = [], None
    for (nome, ocultas), lr, bs in itertools.product(arquiteturas.items(), taxas, batches):
        r = treinar(latente, ocultas, lr, bs, Xtr, Xval, max_epocas, paciencia)
        print(f"  z={latente} arq={nome:12s} lr={lr:<6g} batch={bs:<4d} "
              f"custo_val={r['val']:.2f}  (melhor época {r['ep_melhor']}/{r['ep_rodadas']}, "f"{r['tempo']:.0f}s)")
        linhas.append(dict(latente=latente, arquitetura=nome, lr=lr, batch=bs, custo_val=r["val"], rec_val=r["val_rec"], kl_val=r["val_kl"],
                           epoca_melhor=r["ep_melhor"], epocas_rodadas=r["ep_rodadas"], tempo_s=r["tempo"]))
        if melhor is None or r["val"] < melhor["val"]:
            r["config"] = dict(arquitetura=nome, ocultas=ocultas, lr=lr, batch=bs)
            melhor = r
    return melhor, pd.DataFrame(linhas)


#4 - Espaço Latente (PCA, codificação, métricas de cluster)
@torch.no_grad()
def codificar(modelo, X, bloco=5000):
    """Representação latente = média mu de q(z|x) (sem ruído)."""
    modelo.eval()
    return torch.cat([modelo.codificar(X[i:i + bloco])[0] for i in range(0, len(X), bloco)]).cpu().numpy()
 
 
def metricas_latente(nome, Z_tr, y_tr, Z_te, y_te):
    """Quantifica clusters e separação de rótulos nas coordenadas Z."""
    sil = silhouette_score(Z_te, y_te, sample_size=min(5000, len(Z_te)), random_state=SEED)
    km = KMeans(10, n_init=10, random_state=SEED).fit(Z_te)
    nmi = normalized_mutual_info_score(y_te, km.labels_)
    acc = KNeighborsClassifier(5).fit(Z_tr, y_tr).score(Z_te, y_te)
    return dict(representacao=nome, silhouette_rotulos=sil, NMI_kmeans=nmi, acuracia_knn=acc, dim_ativas=int((Z_te.var(0) > 0.01).sum()))


#5 - Geração de imagens novas (amostrando z ~ N(0,I))
def dispersao(ax, Z, y, titulo, xlabel, ylabel):
    cmap = plt.get_cmap("tab10")
    for c in range(10):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.5, color=cmap(c), label=NOMES[c])
    for c in range(10):  # nome da classe no centro (mediana) dos pontos dela
        m = y == c
        ax.text(np.median(Z[m, 0]), np.median(Z[m, 1]), NOMES[c], fontsize=8, weight="bold",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, lw=0))
    ax.set_title(titulo); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(markerscale=4, fontsize=8, loc="best")
 
 
def mosaico(imgs, linhas, colunas):
    """Junta imagens (N,784) numa grade única."""
    imgs = np.asarray(imgs).reshape(linhas, colunas, 28, 28)
    return imgs.transpose(0, 2, 1, 3).reshape(linhas * 28, colunas * 28)


@torch.no_grad()
def decodificar(modelo, Z):
    modelo.eval()
    return torch.sigmoid(modelo.dec(torch.as_tensor(Z, dtype=torch.float32, device=DISPOSITIVO))).cpu().numpy()
 
 
def figuras(modelos, resultados, Z, y_te, Xte, df_busca, pca8, Z8_pca_te):
    # --- 1) busca de hiperparâmetros ---
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for a, d in zip(ax, LATENTES):
        s = df_busca[df_busca.latente == d].sort_values("custo_val", ascending=False)
        rot = [f"{r.arquitetura} | lr={r.lr:g} | bs={r.batch}" for r in s.itertuples()]
        a.barh(range(len(s)), s.custo_val, color="tab:blue" if d == 2 else "tab:orange")
        a.set_yticks(range(len(s))); a.set_yticklabels(rot, fontsize=7)
        a.set_xlim(s.custo_val.min() * 0.97, s.custo_val.max() * 1.01)
        a.set_title(f"Latente {d}D: custo de validação (menor = melhor)")
        a.set_xlabel("custo por amostra (reconstrução + KL)")
    plt.tight_layout(); plt.savefig(f"{PASTA}/busca_hiperparametros.png", dpi=110); plt.close()
 
    # --- 2) curvas de treino/validação ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, d in zip(ax, LATENTES):
        h = pd.DataFrame(resultados[d]["hist"])
        a.plot(h.epoca, h.treino, label="treino"); a.plot(h.epoca, h.validacao, label="validação")
        a.axvline(resultados[d]["ep_melhor"], ls="--", color="gray", label="melhor época")
        a.set_title(f"Latente {d}D ({resultados[d]['config']['arquitetura']})")
        a.set_xlabel("época"); a.set_ylabel("custo por amostra"); a.legend()
    plt.tight_layout(); plt.savefig(f"{PASTA}/curvas_treino.png", dpi=110); plt.close()
 
    # --- 3) latente 2D ---
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    dispersao(ax, Z[2], y_te, "Espaço latente 2D (conjunto de teste)", "z1", "z2")
    plt.tight_layout(); plt.savefig(f"{PASTA}/latente_2d.png", dpi=120); plt.close()
 
    # --- 4) latente 8D projetado por PCA + variância explicada ---
    var = pca8.explained_variance_ratio_
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    dispersao(ax[0], Z8_pca_te, y_te, f"Latente 8D projetado em 2D (PCA): {100*var[:2].sum():.1f}% da variância",
              f"PC1 ({100*var[0]:.1f}%)", f"PC2 ({100*var[1]:.1f}%)")
    ax[1].bar(range(1, 9), 100 * var, label="por componente")
    ax[1].plot(range(1, 9), 100 * np.cumsum(var), "o-", color="tab:red", label="acumulada")
    ax[1].set_xlabel("componente principal"); ax[1].set_ylabel("variância explicada (%)")
    ax[1].set_title("Variância explicada pelo PCA (8D)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{PASTA}/latente_8d_pca.png", dpi=120); plt.close()
 
    # --- 5) reconstruções: um exemplo de cada classe ---
    idx = [int(np.flatnonzero(y_te == c)[0]) for c in range(10)]
    orig = Xte[idx].cpu().numpy()
    linhas = [("Original", orig)] + [(f"VAE {d}D", decodificar(modelos[d], Z[d][idx])) for d in LATENTES]
    fig, ax = plt.subplots(len(linhas), 1, figsize=(12, 1.4 * len(linhas) + 0.6))
    for a, (nome, im) in zip(ax, linhas):
        a.imshow(mosaico(im, 1, 10), cmap="gray"); a.set_ylabel(nome, rotation=0, ha="right", va="center")
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(f"{PASTA}/reconstrucoes.png", dpi=120); plt.close()
 
    # --- 6) geração: z ~ N(0, I) ---
    g = np.random.default_rng(SEED)
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.6))
    for a, d in zip(ax, LATENTES):
        im = decodificar(modelos[d], g.standard_normal((40, d)))
        a.imshow(mosaico(im, 4, 10), cmap="gray"); a.set_title(f"Amostras novas, VAE {d}D (z ~ N(0,I))")
        a.axis("off")
    plt.tight_layout(); plt.savefig(f"{PASTA}/geracao.png", dpi=120); plt.close()
 
    # --- 7) variedade 2D: decodifica uma grade de z ---
    n = 20
    eixo = np.linspace(-3, 3, n)
    grade = np.array([[x, yv] for yv in eixo[::-1] for x in eixo])  # z2 de cima (3) para baixo (-3)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mosaico(decodificar(modelos[2], grade), n, n), cmap="gray", extent=[-3, 3, -3, 3])
    ax.set_xlabel("z1"); ax.set_ylabel("z2"); ax.set_title("Imagens geradas varrendo o espaço latente 2D")
    plt.tight_layout(); plt.savefig(f"{PASTA}/variedade_2d.png", dpi=120); plt.close()


#Programa principal
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="busca reduzida (teste rápido)")
    ap.add_argument("--sintetico", action="store_true", help="dados falsos, só para testar o código")
    ap.add_argument("--epocas", type=int, default=40, help="máximo de épocas por treino")
    ap.add_argument("--paciencia", type=int, default=5, help="early stopping")
    ap.add_argument("--refinar", type=int, default=0, metavar="N",
                    help="não refaz a busca: retreina só os melhores modelos já salvos, com até N épocas")
    args = ap.parse_args()
 
    arqs, taxas, batches, max_ep = ARQUITETURAS, TAXAS, BATCHES, args.epocas
    if args.rapido:
        arqs = dict(list(ARQUITETURAS.items())[:2]); taxas = TAXAS[:1]; batches = BATCHES[:1]
        max_ep = min(max_ep, 12)
    if args.refinar:
        PASTA = "resultados_vae_refinado"  #não sobrescreve os resultados da busca original
    os.makedirs(PASTA, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Dispositivo: {DISPOSITIVO}")
 
    X, y, Xte, yte = carregar_dados(args.sintetico)
    Xtr, ytr, Xval, yval = dividir(X, y)
    print(f"Treino: {len(Xtr)} | Validação: {len(Xval)} | Teste: {len(Xte)}")
    Xtr_d, Xval_d, Xte_d = Xtr.to(DISPOSITIVO), Xval.to(DISPOSITIVO), Xte.to(DISPOSITIVO)
    ytr, yte_np = ytr.numpy(), yte.numpy()
 
    resultados, modelos = {}, {}
    if args.refinar:
        # ---- MODO REFINAR: pula a busca e retreina só o melhor modelo de cada latente ----
        df_busca = pd.read_csv(f"{PASTA_BUSCA}/busca_hiperparametros.csv")
        df_busca.to_csv(f"{PASTA}/busca_hiperparametros.csv", index=False)  #a busca original continua valendo
        for d in LATENTES:
            ck = torch.load(f"{PASTA_BUSCA}/vae_{d}d.pt", map_location="cpu", weights_only=False)
            c = ck["config"]  #melhor configuração encontrada na busca (arquitetura, lr, batch)
            print(f"\n=== Refinando latente {d}D: {c['arquitetura']} lr={c['lr']:g} batch={c['batch']} "
                  f"(até {args.refinar} épocas, paciência {args.paciencia}) ===")
            r = treinar(d, c["ocultas"], c["lr"], c["batch"], Xtr_d, Xval_d, args.refinar, args.paciencia)
            r["config"] = c
            ant = df_busca[(df_busca.latente == d) & (df_busca.arquitetura == c["arquitetura"]) &
                           np.isclose(df_busca.lr, c["lr"]) & (df_busca.batch == c["batch"])].custo_val.iloc[0]
            r["val_busca"] = float(ant)
            print(f"  custo_val: {ant:.2f} (na busca) -> {r['val']:.2f}  "
                  f"(melhor época {r['ep_melhor']}/{r['ep_rodadas']}, {r['tempo']:.0f}s)")
            resultados[d], modelos[d] = r, r["modelo"]
            torch.save(dict(estado=r["modelo"].state_dict(), config=c, latente=d), f"{PASTA}/vae_{d}d.pt")
    else:
        # ---- busca + melhor modelo de cada latente ----
        buscas = []
        for d in LATENTES:
            print(f"\n=== Busca de hiperparâmetros: latente {d}D ===")
            r, df = buscar(d, Xtr_d, Xval_d, arqs, taxas, batches, max_ep, args.paciencia)
            resultados[d], modelos[d] = r, r["modelo"]
            buscas.append(df)
            print(f"  >>> MELHOR {d}D: {r['config']['arquitetura']} lr={r['config']['lr']:g} "
                  f"batch={r['config']['batch']}  custo_val={r['val']:.2f}")
            torch.save(dict(estado=r["modelo"].state_dict(), config=r["config"], latente=d), f"{PASTA}/vae_{d}d.pt")
        df_busca = pd.concat(buscas, ignore_index=True)
        df_busca.to_csv(f"{PASTA}/busca_hiperparametros.csv", index=False)
 
    # ---- avaliação final no TESTE + codificação ----
    resumo, Z_tr, Z_te = [], {}, {}
    for d in LATENTES:
        t, rc, kl = avaliar(modelos[d], Xte_d)
        c = resultados[d]["config"]
        resumo.append(dict(latente=d, arquitetura=c["arquitetura"], lr=c["lr"], batch=c["batch"],
                           custo_val=resultados[d]["val"], custo_teste=t, rec_teste=rc, kl_teste=kl,
                           epoca_melhor=resultados[d]["ep_melhor"], epocas_rodadas=resultados[d]["ep_rodadas"],
                           **({"custo_val_busca": resultados[d]["val_busca"]} if args.refinar else {})))
        Z_tr[d], Z_te[d] = codificar(modelos[d], Xtr_d), codificar(modelos[d], Xte_d)
    pd.DataFrame(resumo).to_csv(f"{PASTA}/resumo_final.csv", index=False)
 
    # ---- PCA do latente 8D (ajustado no treino, aplicado no teste) ----
    pca8 = PCA(n_components=8).fit(Z_tr[8])
    pd.DataFrame(dict(componente=range(1, 9), variancia=pca8.explained_variance_ratio_,
                      acumulada=np.cumsum(pca8.explained_variance_ratio_))).to_csv(f"{PASTA}/variancia_pca_8d.csv", index=False)
    P_tr, P_te = pca8.transform(Z_tr[8])[:, :2], pca8.transform(Z_te[8])[:, :2]
 
    # ---- métricas de clusters / separação ----
    met = [metricas_latente("Latente 2D", Z_tr[2], ytr, Z_te[2], yte_np),
           metricas_latente("Latente 8D (completo)", Z_tr[8], ytr, Z_te[8], yte_np),
           metricas_latente("Latente 8D -> PCA 2D", P_tr, ytr, P_te, yte_np)]
    df_met = pd.DataFrame(met)
    df_met.to_csv(f"{PASTA}/metricas_latente.csv", index=False)
 
    figuras(modelos, resultados, Z_te, yte_np, Xte_d, df_busca, pca8, P_te)
 
    # ---- resumo no terminal ----
    print("\n===== MELHORES MODELOS (custo por amostra = reconstrução + KL) =====")
    print(pd.DataFrame(resumo).round({"custo_val": 2, "custo_teste": 2, "rec_teste": 2, "kl_teste": 2, "custo_val_busca": 2}).to_string(index=False))
    var = pca8.explained_variance_ratio_
    print(f"\n===== PCA do latente 8D: PC1={100*var[0]:.1f}%  PC2={100*var[1]:.1f}%  "
          f"PC1+PC2={100*var[:2].sum():.1f}%  (8 componentes: {100*var.sum():.0f}%) =====")
    print("\n===== CLUSTERS / SEPARAÇÃO DOS RÓTULOS NO ESPAÇO LATENTE =====")
    print(df_met.round(3).to_string(index=False))
    print(f"\nArquivos salvos em: {os.path.abspath(PASTA)}")