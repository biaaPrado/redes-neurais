"""
QUANTIZAÇÃO DE CORES COM MiniSom (SOM) E GNG  -  versão experimental
====================================================================
Uso: python quantizacao_final.py imagens/

Dependências:
    pip install numpy pandas matplotlib pillow minisom scikit-image

Saídas (pasta resultados/):
    resultados_completos.csv   -> TODAS as simulações (imagem x rede x hiperparâmetros)
    melhores_por_imagem.csv    -> melhor configuração de cada (imagem, rede)
    resumo_por_rede.csv        -> média/desvio das 5+ imagens, por rede
    img_<nome>.png             -> avaliação qualitativa (reconstruções + mapas de erro)
    resumo_metricas.png        -> comparação quantitativa SOM x GNG
    hiperparametros.png        -> efeito de cada configuração de hiperparâmetros
"""
import os
import sys
import glob
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from minisom import MiniSom
from skimage.metrics import structural_similarity


# Arquitetura fixa, apenas hiperparâmetros variam (para estudo de sensibilidade)
SEED = 42
LADO_MENOR = 128  # imagens maiores têm o lado menor reduzido p/ 128 px (treino mais rápido)
LADO_MINIMO = 100  # exigência do trabalho

REDES_SOM = [4, 8, 16]  # grades 4x4, 8x8, 16x16  (16, 64 e 256 neurônios)
REDES_GNG = [16, 64, 256]  # nº máximo de neurônios do GNG

# hiperparâmetros do SOM (MiniSom). sigma = fração * lado da grade
GRADE_SOM = dict(sigma_frac=[0.5, 0.25], lr=[0.1, 0.5, 1.0])
N_ITER_SOM = 20000

# hiperparâmetros do GNG (lam, alpha e d ficam fixos nos valores clássicos)
GRADE_GNG = dict(eps_b=[0.05, 0.2], eps_n=[0.001, 0.006], idade_max=[25, 50])
LAM, ALPHA, D = 100, 0.5, 0.995


# 1) EXTRAÇÃO + NORMALIZAÇÃO MIN-MAX
def carregar_imagem(caminho):
    im = Image.open(caminho).convert("RGB")
    w, h = im.size
    if min(w, h) < LADO_MINIMO:
        raise ValueError(f"{caminho}: {w}x{h} é menor que {LADO_MINIMO}x{LADO_MINIMO}")
    if min(w, h) > LADO_MENOR:  # reduz mantendo a proporção
        f = LADO_MENOR / min(w, h)
        im = im.resize((round(w * f), round(h * f)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0  # valores em [0,1] (só p/ guardar)


def normalizar_minmax(img):
    """x' = (x - min) / (max - min), calculado separadamente em cada canal R, G, B."""
    mn = img.min(axis=(0, 1))
    mx = img.max(axis=(0, 1))
    return (img - mn) / (mx - mn + 1e-12), mn, mx


def desnormalizar(img_norm, mn, mx):
    """Operação inversa, só para EXIBIR a imagem com as cores originais."""
    return np.clip(img_norm * (mx - mn) + mn, 0, 1)


# 2) FUNÇÕES AUXILIARES
def dois_melhores(W, X, bloco=20000):
    """Para cada pixel: índice do neurônio vencedor (BMU) e do segundo colocado."""
    b1 = np.empty(len(X), dtype=np.int64)
    b2 = np.empty(len(X), dtype=np.int64)
    w2 = (W ** 2).sum(1)
    for i in range(0, len(X), bloco):
        x = X[i:i + bloco]
        d = w2[None, :] - 2 * x @ W.T  # |x|² omitido (não altera o argmin)
        idx = np.argpartition(d, 1, axis=1)[:, :2]
        ordem = np.take_along_axis(d, idx, 1).argsort(1)
        idx = np.take_along_axis(idx, ordem, 1)
        b1[i:i + bloco], b2[i:i + bloco] = idx[:, 0], idx[:, 1]
    return b1, b2


# 3) SOM (biblioteca MiniSom)
def treinar_som(X, lado, sigma_frac, lr, seed):
    sigma = max(1.0, lado * sigma_frac)
    som = MiniSom(lado, lado, 3, sigma=sigma, learning_rate=lr, neighborhood_function="gaussian", random_seed=seed)
    som.random_weights_init(X)  # pesos iniciais = pixels sorteados
    som.train_random(X, N_ITER_SOM)  # treino online
    W = som.get_weights().reshape(-1, 3)  # (lado*lado, 3); índice k = linha*lado + coluna

    def erro_topologico(b1, b2):
        # 1º e 2º BMU devem ser vizinhos na GRADE (distância de Chebyshev <= 1)
        l1, c1 = np.divmod(b1, lado)
        l2, c2 = np.divmod(b2, lado)
        return (np.maximum(abs(l1 - l2), abs(c1 - c2)) > 1).mean()

    return W, erro_topologico


# 4) GNG (implementação própria)
class GNG:
    def __init__(self, max_nos, X, seed):
        self.rng = np.random.default_rng(seed)
        self.max_nos = max_nos
        self.W = np.zeros((max_nos, 3))
        self.erro = np.zeros(max_nos)
        self.ativo = np.zeros(max_nos, bool)
        self.idade = -np.ones((max_nos, max_nos), dtype=np.int32)  # -1 = sem aresta
        self.W[:2] = X[self.rng.choice(len(X), 2, replace=False)]
        self.ativo[:2] = True

    def treinar(self, X, eps_b, eps_n, idade_max, lam=LAM, alpha=ALPHA, d=D, passos_extras=15000):
        n_passos = lam * self.max_nos + passos_extras  # tempo p/ crescer + ajuste fino
        amostras = X[self.rng.integers(0, len(X), n_passos)]
        for t, x in enumerate(amostras, start=1):
            ids = np.flatnonzero(self.ativo)
            dist = ((self.W[ids] - x) ** 2).sum(1)
            o = np.argsort(dist)[:2]
            s1, s2 = ids[o[0]], ids[o[1]]
            self.erro[s1] += dist[o[0]]
            viz = np.flatnonzero(self.idade[s1] >= 0)
            self.W[s1] += eps_b * (x - self.W[s1])
            self.W[viz] += eps_n * (x - self.W[viz])
            self.idade[s1, viz] += 1
            self.idade[viz, s1] += 1
            self.idade[s1, s2] = self.idade[s2, s1] = 0
            velhas = np.flatnonzero(self.idade[s1] > idade_max)
            self.idade[s1, velhas] = -1
            self.idade[velhas, s1] = -1
            for k in np.append(velhas, s1):
                if self.ativo[k] and not (self.idade[k] >= 0).any():
                    self.ativo[k] = False
            if t % lam == 0 and self.ativo.sum() < self.max_nos:
                self._inserir(alpha)
            self.erro[self.ativo] *= d
        return self

    def _inserir(self, alpha):
        ids = np.flatnonzero(self.ativo)
        q = ids[np.argmax(self.erro[ids])]
        viz = np.flatnonzero(self.idade[q] >= 0)
        if len(viz) == 0:
            return
        f = viz[np.argmax(self.erro[viz])]
        r = np.flatnonzero(~self.ativo)[0]
        self.W[r] = (self.W[q] + self.W[f]) / 2
        self.ativo[r] = True
        self.idade[q, f] = self.idade[f, q] = -1
        self.idade[q, r] = self.idade[r, q] = 0
        self.idade[f, r] = self.idade[r, f] = 0
        self.erro[q] *= alpha
        self.erro[f] *= alpha
        self.erro[r] = self.erro[q]


def treinar_gng(X, n_neuronios, eps_b, eps_n, idade_max, seed):
    g = GNG(n_neuronios, X, seed).treinar(X, eps_b, eps_n, idade_max)
    ids = np.flatnonzero(g.ativo)
    W = g.W[ids]

    def erro_topologico(b1, b2):
        # 1º e 2º BMU devem estar LIGADOS por uma aresta do grafo
        return (g.idade[ids[b1], ids[b2]] < 0).mean()

    return W, erro_topologico


# 5) INFERÊNCIA + MÉTRICAS
def avaliar(W, erro_topologico, X, img_norm):
    b1, b2 = dois_melhores(W, X)  # BMU de cada pixel
    Xq = W[b1]  # troca a cor pelo peso do BMU
    img_q = Xq.reshape(img_norm.shape)  # reconstrução da matriz
    mse = ((X - Xq) ** 2).mean()
    return dict(
        QE=np.linalg.norm(X - Xq, axis=1).mean(),  # erro de quantização
        TE=erro_topologico(b1, b2),  # erro topológico
        MSE=mse,
        PSNR=10 * np.log10(1.0 / mse),  # dB (dados em [0,1])
        SSIM=structural_similarity(img_norm, img_q, channel_axis=2, data_range=1.0),
        cores_usadas=len(np.unique(b1)),
    ), img_q


def rodar_imagem(nome, img_norm):
    X = img_norm.reshape(-1, 3)
    linhas, guardadas = [], {}
    # ---- SOM ----
    for lado, sf, lr in itertools.product(REDES_SOM, GRADE_SOM["sigma_frac"], GRADE_SOM["lr"]):
        t0 = time.time()
        W, te = treinar_som(X, lado, sf, lr, SEED)
        m, img_q = avaliar(W, te, X, img_norm)
        hp = f"sigma={max(1.0, lado*sf):g}, lr={lr}"
        linhas.append(dict(imagem=nome, modelo="SOM", neuronios=lado * lado, hiperparametros=hp, tempo_s=time.time() - t0, **m))
        guardadas[("SOM", lado * lado, hp)] = img_q
    # ---- GNG ----
    for n, eb, en, ia in itertools.product(REDES_GNG, GRADE_GNG["eps_b"], GRADE_GNG["eps_n"], GRADE_GNG["idade_max"]):
        t0 = time.time()
        W, te = treinar_gng(X, n, eb, en, ia, SEED)
        m, img_q = avaliar(W, te, X, img_norm)
        hp = f"eps_b={eb}, eps_n={en}, idade_max={ia}"
        linhas.append(dict(imagem=nome, modelo="GNG", neuronios=n, hiperparametros=hp, tempo_s=time.time() - t0, **m))
        guardadas[("GNG", n, hp)] = img_q
    return linhas, guardadas


# 6) AVALIAÇÃO QUALITATIVA E QUANTITATIVA (figuras)
def figura_qualitativa(nome, img, img_norm, mn, mx, df_img, guardadas):
    """Linha 1: reconstruções (melhor config de cada rede). Linha 2: mapa de erro."""
    fig, ax = plt.subplots(2, 7, figsize=(24, 7))
    ax[0, 0].imshow(img); ax[0, 0].set_title("Original")
    ax[1, 0].axis("off")
    col = 1
    for modelo in ("SOM", "GNG"):
        for n in (16, 64, 256):
            sub = df_img[(df_img.modelo == modelo) & (df_img.neuronios == n)]
            b = sub.loc[sub.QE.idxmin()]
            img_q = guardadas[(modelo, n, b.hiperparametros)]
            ax[0, col].imshow(desnormalizar(img_q, mn, mx))
            ax[0, col].set_title(f"{modelo} {n} cores\nQE={b.QE:.3f}  PSNR={b.PSNR:.1f} dB\n" f"SSIM={b.SSIM:.3f}  TE={b.TE:.3f}", fontsize=9)
            erro = np.abs(img_norm - img_q).mean(axis=2)  # erro médio dos 3 canais
            ax[1, col].imshow(erro, cmap="inferno", vmin=0, vmax=0.5)
            ax[1, col].set_title("mapa de erro", fontsize=9)
            col += 1
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.suptitle(f"Imagem: {nome}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"resultados/img_{nome}.png", dpi=110)
    plt.close()


def figuras_resumo(df, melhores):
    # --- comparação quantitativa SOM x GNG (média ± desvio entre imagens) ---
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for a, met in zip(ax, ("QE", "TE", "PSNR", "SSIM")):
        for modelo, cor in (("SOM", "tab:blue"), ("GNG", "tab:orange")):
            g = melhores[melhores.modelo == modelo].groupby("neuronios")[met]
            a.errorbar(g.mean().index, g.mean().values, yerr=g.std().values, fmt="o-",
                       capsize=4, label=modelo, color=cor)
        a.set_xscale("log", base=2); a.set_xlabel("nº de neurônios (cores)")
        a.set_title(f"{met} (melhor config, média ± dp)"); a.legend()
    plt.tight_layout(); plt.savefig("resultados/resumo_metricas.png", dpi=110); plt.close()

    # --- efeito dos hiperparâmetros (QE médio entre as imagens) ---
    fig, ax = plt.subplots(2, 3, figsize=(20, 9))
    for i, modelo in enumerate(("SOM", "GNG")):
        for j, n in enumerate((16, 64, 256)):
            s = df[(df.modelo == modelo) & (df.neuronios == n)].groupby("hiperparametros").QE
            m, d = s.mean(), s.std()
            ax[i, j].barh(range(len(m)), m.values, xerr=d.values, color="tab:blue" if i == 0 else "tab:orange")
            ax[i, j].set_yticks(range(len(m))); ax[i, j].set_yticklabels(m.index, fontsize=7)
            ax[i, j].set_title(f"{modelo} - {n} neurônios"); ax[i, j].set_xlabel("QE médio")
    plt.tight_layout(); plt.savefig("resultados/hiperparametros.png", dpi=110); plt.close()


def figura_painel(nome, img, mn, mx, df_img, guardadas):
    """Uma figura por imagem (para o relatório): Original no topo; abaixo, uma linha por K
    (16, 64, 256) com GNG à esquerda e SOM à direita. Usa a melhor config (menor QE) de cada rede.
    Salva em resultados/paineis/<nome>_painel.png"""
    pasta = "resultados/paineis"
    os.makedirs(pasta, exist_ok=True)
    fig = plt.figure(figsize=(8, 10.2), facecolor="white")
    gs = fig.add_gridspec(4, 4, hspace=0.30, wspace=0.08, left=0.02, right=0.98, top=0.965, bottom=0.015)
    ax = fig.add_subplot(gs[0, 1:3])  # original centralizada
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Imagem original", fontsize=11, fontweight="bold")
    for linha, n in enumerate((16, 64, 256), start=1):
        for coluna, modelo in ((slice(0, 2), "GNG"), (slice(2, 4), "SOM")):
            sub = df_img[(df_img.modelo == modelo) & (df_img.neuronios == n)]
            b = sub.loc[sub.QE.idxmin()]
            img_q = desnormalizar(guardadas[(modelo, n, b.hiperparametros)], mn, mx)
            ax = fig.add_subplot(gs[linha, coluna])
            ax.imshow(img_q); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{modelo}, K={n}\nQE={b.QE:.3f} | PSNR={b.PSNR:.1f} dB | SSIM={b.SSIM:.3f}", fontsize=9)
    fig.savefig(f"{pasta}/{nome}_painel.png", dpi=150, facecolor="white")
    plt.close(fig)


def figura_estilo_referencia(nome, img, mn, mx, df_img, guardadas):
    """Uma figura por rede: 'Original' x 'Quantized (K=...)' + paleta de centroides.
    Usa a melhor configuração (menor QE) de cada rede. Salva em resultados/estilo_referencia/."""
    from matplotlib.patches import Rectangle
    pasta = "resultados/estilo_referencia"
    os.makedirs(pasta, exist_ok=True)
    FUNDO, TEXTO = "#111111", "white"
    for modelo in ("SOM", "GNG"):
        for n in (16, 64, 256):
            sub = df_img[(df_img.modelo == modelo) & (df_img.neuronios == n)]
            b = sub.loc[sub.QE.idxmin()]
            img_q = desnormalizar(guardadas[(modelo, n, b.hiperparametros)], mn, mx)

            # centroides = cores que a rede realmente usou na imagem (únicas), ordenadas por brilho
            paleta = np.unique(img_q.reshape(-1, 3), axis=0)
            paleta = paleta[np.argsort(paleta @ [0.299, 0.587, 0.114])]
            por_linha = 16
            n_lin = int(np.ceil(len(paleta) / por_linha))

            fig = plt.figure(figsize=(9, 4.2 + 0.32 * n_lin), facecolor=FUNDO)
            gs = fig.add_gridspec(2, 2, height_ratios=[6, 0.5 + 0.55 * n_lin], hspace=0.12, wspace=0.04, left=0.03, right=0.97,
                                  top=0.93, bottom=0.03)
            for col, (im, tit) in enumerate(((img, "Original"), (img_q, f"Quantized ({modelo}, K={n})"))):
                ax = fig.add_subplot(gs[0, col])
                ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(tit, color=TEXTO, fontsize=12, fontweight="bold")
                for s in ax.spines.values():
                    s.set_color("#555555")

            ax = fig.add_subplot(gs[1, :])
            ax.set_facecolor(FUNDO)
            ax.set_xlim(0, por_linha); ax.set_ylim(n_lin, 0); ax.axis("off")
            for i, cor in enumerate(paleta):
                r, c = divmod(i, por_linha)
                ax.add_patch(Rectangle((c + 0.06, r + 0.08), 0.88, 0.84, color=cor))
            ax.set_title(f"Extracted Color Palette Centroids ({len(paleta)} cores usadas):", color=TEXTO, fontsize=10, loc="left")
            fig.savefig(f"{pasta}/{nome}_{modelo}_K{n}.png", dpi=130, facecolor=FUNDO)
            plt.close(fig)





if __name__ == "__main__":
    print("Executando o arquivo:", os.path.abspath(__file__))  # confirma QUAL arquivo está rodando
    pasta = sys.argv[1] if len(sys.argv) > 1 else "imagens"
    arquivos = sorted(f for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
                      for f in glob.glob(os.path.join(pasta, ext)))
    if len(arquivos) < 5:
        print(f"AVISO: encontrei {len(arquivos)} imagem(ns) em '{pasta}'; o trabalho pede >= 5.")
    if not arquivos:
        sys.exit("Nenhuma imagem encontrada.")
    os.makedirs("resultados", exist_ok=True)

    todas = []
    for arq in arquivos:
        nome = os.path.splitext(os.path.basename(arq))[0]
        img = carregar_imagem(arq)
        img_norm, mn, mx = normalizar_minmax(img)
        print(f"\n=== {nome}: {img.shape[1]}x{img.shape[0]} px ===")
        linhas, guardadas = rodar_imagem(nome, img_norm)
        df_img = pd.DataFrame(linhas)
        for _, b in df_img.groupby(["modelo", "neuronios"]).QE.idxmin().items():
            r = df_img.loc[b]
            print(f"  melhor {r.modelo} {r.neuronios:3d}: QE={r.QE:.4f} TE={r.TE:.3f} "
                  f"PSNR={r.PSNR:.1f} SSIM={r.SSIM:.3f}  [{r.hiperparametros}]")
        figura_qualitativa(nome, img, img_norm, mn, mx, df_img, guardadas)
        figura_estilo_referencia(nome, img, mn, mx, df_img, guardadas)
        figura_painel(nome, img, mn, mx, df_img, guardadas)
        todas.append(df_img)

    df = pd.concat(todas, ignore_index=True)
    df.to_csv("resultados/resultados_completos.csv", index=False)
    idx = df.groupby(["imagem", "modelo", "neuronios"]).QE.idxmin()
    melhores = df.loc[idx].reset_index(drop=True)
    melhores.to_csv("resultados/melhores_por_imagem.csv", index=False)
    resumo = melhores.groupby(["modelo", "neuronios"])[["QE", "TE", "MSE", "PSNR", "SSIM"]].agg(["mean", "std"]).round(4)
    resumo.to_csv("resultados/resumo_por_rede.csv")
    figuras_resumo(df, melhores)
    print("\n===== RESUMO (média entre as imagens, melhor config de cada uma) =====")
    print(resumo.to_string())
    print(f"\nTotal de simulações: {len(df)}  ->  pasta 'resultados/'")
    print("Painéis (relatório) salvos em:", os.path.abspath("resultados/paineis"))
    print("Figuras estilo referência em:", os.path.abspath("resultados/estilo_referencia"))