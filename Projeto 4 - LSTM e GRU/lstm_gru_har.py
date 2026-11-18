"""
CLASSIFICAÇÃO DE SÉRIES TEMPORAIS COM LSTM / GRU  -  Reconhecimento de Atividade Humana (UCI HAR)
================================================================================================
Tarefa: dada uma janela de 2,56 s (128 leituras a 50 Hz) de acelerômetro e giroscópio de um smartphone preso à cintura, 
classificar a atividade: andando, subindo escada, descendo escada, sentado, em pé ou deitado.

Uso:
    python lstm_gru_har.py                 # busca completa
    python lstm_gru_har.py --rapido        # versão reduzida, para testar se tudo funciona

Dados (UCI Machine Learning Repository, id 240, licença CC BY 4.0): https://archive.ics.uci.edu/dataset/240

Dependências:
    pip install torch numpy pandas matplotlib

Saídas (pasta resultados_har/):
    busca_hiperparametros.csv   -> todas as configurações testadas
    efeito_hiperparametros.csv  -> efeito médio de cada hiperparâmetro (custo e acurácia de validação)
    metricas_teste.csv          -> LSTM e GRU no teste (média e desvio entre sementes) + baseline
    metricas_por_classe.csv     -> precisão, revocação e F1 de cada classe (LSTM e GRU)
    exemplos_sinais.png         -> um exemplo de janela de cada atividade
    busca_hiperparametros.png   -> custo de validação de cada configuração
    efeito_hiperparametros.png  -> efeito médio de cada hiperparâmetro
    curvas_aprendizagem.png     -> curvas de custo e acurácia (treino e validação) de LSTM e GRU
    matriz_confusao.png         -> matriz de confusão do melhor modelo no teste
    f1_por_classe.png           -> F1 por classe: LSTM x GRU
"""
import os
import sys
import time
import copy
import zipfile
import argparse
import itertools
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#Configurações globais
SEED = 42
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASTA = "resultados_har"
URL_HAR = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
CLASSES = ["Andando", "Subindo escada", "Descendo escada", "Sentado", "Em pé", "Deitado"]
CANAIS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
          "total_acc_x", "total_acc_y", "total_acc_z"]
N_SUJEITOS_VAL = 4  # dos 21 voluntários do treino oficial, 4 viram validação

#Espaço de busca
GRADE = dict(
    tipo=["LSTM", "GRU"],
    hidden=[32, 64, 128],  # neurônios por camada recorrente
    camadas=[1, 2],  # camadas recorrentes empilhadas
    lr=[1e-3, 5e-3],  # taxa de aprendizado do Adam
)
DROPOUT = 0.3  # entre camadas recorrentes e antes da camada de saída
BATCH = 64
MAX_EPOCAS = 40
PACIENCIA = 6  # early stopping
N_SEMENTES_FINAIS = 3


# Carregamento de dados
def baixar_har(pasta="dados"):
    """Garante que dados/UCI HAR Dataset/ existe: se faltar, baixa o .zip e extrai."""
    raiz = os.path.join(pasta, "UCI HAR Dataset")
    if os.path.isdir(raiz):
        return raiz
    os.makedirs(pasta, exist_ok=True)
    arq = os.path.join(pasta, "uci_har.zip")
    if not os.path.exists(arq):
        print("Baixando o UCI HAR (≈60 MB)...")
        try:
            urllib.request.urlretrieve(URL_HAR, arq)
        except Exception as e:
            sys.exit(f"Falha no download ({e}). Baixe em https://archive.ics.uci.edu/dataset/240 " f"e salve o .zip como {arq}")
    with zipfile.ZipFile(arq) as z:
        z.extractall(pasta)
    interno = os.path.join(pasta, "UCI HAR Dataset.zip")  # o zip da UCI traz outro zip dentro
    if os.path.exists(interno):
        with zipfile.ZipFile(interno) as z:
            z.extractall(pasta)
    if not os.path.isdir(raiz):
        sys.exit(f"Extraí o zip, mas não achei a pasta '{raiz}'. Descompacte o dataset à mão de modo que " f"exista {raiz}/train e {raiz}/test.")
    return raiz


def carregar_har():
    """Retorna X (n, 128, 9), y (0..5), sujeito (1..30) e se a janela é do teste OFICIAL."""
    raiz = baixar_har()
    Xs, ys, ss, ts = [], [], [], []
    for parte in ["train", "test"]:
        sinais = [pd.read_csv(f"{raiz}/{parte}/Inertial Signals/{c}_{parte}.txt", sep=r"\s+", header=None).values for c in CANAIS]
        Xs.append(np.stack(sinais, axis=-1).astype(np.float32))  # (n, 128, 9)
        ys.append(np.loadtxt(f"{raiz}/{parte}/y_{parte}.txt", dtype=int) - 1)  # rótulos 1..6 -> 0..5
        ss.append(np.loadtxt(f"{raiz}/{parte}/subject_{parte}.txt", dtype=int))
        ts.append(np.full(len(ys[-1]), parte == "test"))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), np.concatenate(ts)


def dividir_por_sujeito(X, y, sujeito, oficial_teste):
    """Treino / validação / teste separados por VOLUNTÁRIO (nunca a mesma pessoa em dois conjuntos)."""
    rng = np.random.default_rng(SEED)
    suj_treino_oficial = np.unique(sujeito[~oficial_teste])
    suj_val = set(rng.choice(suj_treino_oficial, N_SUJEITOS_VAL, replace=False).tolist())
    m_te = oficial_teste
    m_va = ~oficial_teste & np.isin(sujeito, list(suj_val))
    m_tr = ~oficial_teste & ~np.isin(sujeito, list(suj_val))
    mu = X[m_tr].mean(axis=(0, 1))  # padronização por canal, só com estatísticas do TREINO
    sd = X[m_tr].std(axis=(0, 1)) + 1e-8
    d = {}
    for nome, m in [("tr", m_tr), ("va", m_va), ("te", m_te)]:
        d[nome] = dict(X=torch.tensor((X[m] - mu) / sd, dtype=torch.float32, device=DISPOSITIVO),
                       y=torch.tensor(y[m], dtype=torch.long, device=DISPOSITIVO),
                       sujeitos=sorted(set(sujeito[m].tolist())), bruto=X[m], y_np=y[m])
    return d, sorted(suj_val)


#Modelo LSTM / GRU
class Classificador(nn.Module):
    def __init__(self, tipo, n_feat, hidden, camadas, n_classes, dropout=DROPOUT):
        super().__init__()
        rnn = nn.LSTM if tipo == "LSTM" else nn.GRU
        self.rnn = rnn(n_feat, hidden, num_layers=camadas, batch_first=True, dropout=dropout if camadas > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.saida = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: (lote, 128, 9)
        out, _ = self.rnn(x)  # estado oculto em cada instante da janela
        return self.saida(self.drop(out[:, -1, :]))  # classifica a partir do ÚLTIMO instante


@torch.no_grad()
def logits_de(modelo, X, bloco=1024):
    modelo.eval()
    return torch.cat([modelo(X[i:i + bloco]) for i in range(0, len(X), bloco)])


def custo_acc(modelo, X, y):
    lg = logits_de(modelo, X)
    return F.cross_entropy(lg, y).item(), (lg.argmax(1) == y).float().mean().item()


#Treino com early stopping (validação)
def treinar(cfg, d, seed, max_epocas, paciencia):
    torch.manual_seed(seed)
    Xtr, ytr = d["tr"]["X"], d["tr"]["y"]
    modelo = Classificador(cfg["tipo"], Xtr.shape[2], cfg["hidden"], cfg["camadas"], len(CLASSES)).to(DISPOSITIVO)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    n, hist, melhor, estado, ep_melhor, sem_melhora = len(ytr), [], float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(1, max_epocas + 1):
        modelo.train()
        perm = torch.randperm(n, device=DISPOSITIVO)
        soma, acertos = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            lg = modelo(Xtr[idx])
            loss = F.cross_entropy(lg, ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)  # estabiliza o treino de RNNs
            opt.step()
            soma += loss.item() * len(idx)
            acertos += (lg.argmax(1) == ytr[idx]).sum().item()
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
    return dict(modelo=modelo, hist=hist, val_loss=melhor, val_acc=acc_m, ep_melhor=ep_melhor, ep_rodadas=len(hist), tempo=time.time() - t0)


#Métricas de avaliação
def matriz_confusao(y, pred, n=len(CLASSES)):
    m = np.zeros((n, n), dtype=int)
    np.add.at(m, (y, pred), 1)  # linha = classe real, coluna = classe prevista
    return m


def metricas(y, pred):
    cm = matriz_confusao(y, pred)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rev = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rev / np.maximum(prec + rev, 1e-12)
    return dict(acuracia=tp.sum() / cm.sum(), f1_macro=f1.mean()), dict(precisao=prec, revocacao=rev, f1=f1), cm


#Figuras de saída
def figuras(d, df_busca, df_efeito, hist_tipo, cm, nome_cm, f1_tipo):
    # --- exemplos de sinais ---
    fig, ax = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    Xb, yb = d["tr"]["bruto"], d["tr"]["y_np"]
    t = np.arange(128) / 50
    for c, a in enumerate(ax.ravel()):
        w = Xb[np.flatnonzero(yb == c)[0]]
        for k, (rot, cor) in enumerate(zip(["x", "y", "z"], ["tab:red", "tab:green", "tab:blue"])):
            a.plot(t, w[:, 6 + k], color=cor, lw=1, label=f"aceleração total {rot}")
        a.set_title(CLASSES[c]); a.set_xlabel("tempo (s)")
    ax[0, 0].set_ylabel("aceleração (g)"); ax[1, 0].set_ylabel("aceleração (g)"); ax[0, 0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{PASTA}/exemplos_sinais.png", dpi=110); plt.close()

    # --- busca ---
    s = df_busca.sort_values("val_loss", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(s) + 1.5))
    ax.barh(range(len(s)), s.val_loss, color=["tab:blue" if t == "LSTM" else "tab:orange" for t in s.tipo])
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([f"{r.tipo} h={r.hidden} c={r.camadas} lr={r.lr:g}  (acc val={r.val_acc:.3f})"
                        for r in s.itertuples()], fontsize=7)
    ax.set_xlabel("custo de validação (entropia cruzada)")
    ax.set_title("Busca de hiperparâmetros (azul = LSTM, laranja = GRU)")
    plt.tight_layout(); plt.savefig(f"{PASTA}/busca_hiperparametros.png", dpi=110); plt.close()

    # --- efeito de cada hiperparâmetro ---
    nomes = list(GRADE)
    fig, ax = plt.subplots(2, len(nomes), figsize=(3.2 * len(nomes), 6), sharey="row")
    for j, h in enumerate(nomes):
        g = df_efeito[df_efeito.hiperparametro == h]
        ax[0, j].bar([str(v) for v in g.valor], g.val_loss_medio, yerr=g.val_loss_dp, capsize=3, color="tab:gray")
        ax[0, j].set_title(h)
        ax[1, j].bar([str(v) for v in g.valor], g.val_acc_media, yerr=g.val_acc_dp, capsize=3, color="tab:green")
    ax[0, 0].set_ylabel("custo de validação médio"); ax[1, 0].set_ylabel("acurácia de validação média")
    plt.tight_layout(); plt.savefig(f"{PASTA}/efeito_hiperparametros.png", dpi=110); plt.close()

    # --- curvas de aprendizagem ---
    fig, ax = plt.subplots(2, len(hist_tipo), figsize=(6 * len(hist_tipo), 7), squeeze=False)
    for j, (tipo, (h, cfg, ep)) in enumerate(hist_tipo.items()):
        h = pd.DataFrame(h)
        titulo = f"{tipo} (h={cfg['hidden']}, c={cfg['camadas']}, lr={cfg['lr']:g})"
        ax[0, j].plot(h.epoca, h.loss_treino, label="treino"); ax[0, j].plot(h.epoca, h.loss_val, label="validação")
        ax[0, j].set_title(titulo); ax[0, j].set_ylabel("entropia cruzada")
        ax[1, j].plot(h.epoca, h.acc_treino, label="treino"); ax[1, j].plot(h.epoca, h.acc_val, label="validação")
        ax[1, j].set_ylabel("acurácia"); ax[1, j].set_xlabel("época")
        for a in ax[:, j]:
            a.axvline(ep, ls="--", color="gray", label="melhor época"); a.legend()
    plt.tight_layout(); plt.savefig(f"{PASTA}/curvas_aprendizagem.png", dpi=110); plt.close()

    # --- matriz de confusão ---
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    pct = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n{100 * pct[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("classe prevista"); ax.set_ylabel("classe real")
    ax.set_title(f"Matriz de confusão no teste ({nome_cm})"); plt.colorbar(im, label="fração da classe real")
    plt.tight_layout(); plt.savefig(f"{PASTA}/matriz_confusao.png", dpi=120); plt.close()

    # --- F1 por classe ---
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.38
    for k, (tipo, f1) in enumerate(f1_tipo.items()):
        ax.bar(np.arange(len(CLASSES)) + (k - 0.5) * w, f1, w, label=tipo)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_ylabel("F1 no teste"); ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("F1 por classe")
    plt.tight_layout(); plt.savefig(f"{PASTA}/f1_por_classe.png", dpi=110); plt.close()


#Programa principal
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="busca reduzida (teste rápido)")
    ap.add_argument("--epocas", type=int, default=MAX_EPOCAS)
    ap.add_argument("--sementes", type=int, default=N_SEMENTES_FINAIS)
    args = ap.parse_args()

    grade, max_ep, n_sem = dict(GRADE), args.epocas, args.sementes
    if args.rapido:
        grade = dict(tipo=["LSTM", "GRU"], hidden=[32], camadas=[1], lr=[5e-3])
        max_ep, n_sem = min(max_ep, 8), 1
    os.makedirs(PASTA, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Dispositivo: {DISPOSITIVO}")

    X, y, sujeito, oficial_teste = carregar_har()
    d, suj_val = dividir_por_sujeito(X, y, sujeito, oficial_teste)
    print(f"Janelas: {X.shape} (n, instantes, canais) | classes: {len(CLASSES)}")
    for nome, rot in [("tr", "Treino"), ("va", "Validação"), ("te", "Teste")]:
        print(f"{rot:9s}: {len(d[nome]['y']):5d} janelas | voluntários {d[nome]['sujeitos']}")
    prop_classes = np.bincount(d["tr"]["y_np"], minlength=len(CLASSES)) / len(d["tr"]["y_np"])
    print("Proporção das classes no treino:", dict(zip(CLASSES, prop_classes.round(3))))

    # ---- busca de hiperparâmetros (pela VALIDAÇÃO) ----
    linhas, combos = [], list(itertools.product(*grade.values()))
    print(f"\n=== Busca de hiperparâmetros: {len(combos)} configurações ===")
    for i, vals in enumerate(combos, 1):
        cfg = dict(zip(grade, vals))
        r = treinar(cfg, d, SEED, max_ep, PACIENCIA)
        linhas.append(dict(**cfg, val_loss=r["val_loss"], val_acc=r["val_acc"], epoca_melhor=r["ep_melhor"],
                           epocas_rodadas=r["ep_rodadas"], tempo_s=r["tempo"]))
        print(f"  [{i:2d}/{len(combos)}] {cfg['tipo']:4s} h={cfg['hidden']:<3d} c={cfg['camadas']} lr={cfg['lr']:<6g} "
              f"val_loss={r['val_loss']:.4f} val_acc={r['val_acc']:.4f} (época {r['ep_melhor']}/{r['ep_rodadas']}, {r['tempo']:.0f}s)")
    df_busca = pd.DataFrame(linhas)
    df_busca.to_csv(f"{PASTA}/busca_hiperparametros.csv", index=False)
    efeito = [dict(hiperparametro=h, valor=v, val_loss_medio=g.val_loss.mean(), val_loss_dp=g.val_loss.std(),
                   val_acc_media=g.val_acc.mean(), val_acc_dp=g.val_acc.std())
              for h in grade for v, g in df_busca.groupby(h)]
    df_efeito = pd.DataFrame(efeito)
    df_efeito.to_csv(f"{PASTA}/efeito_hiperparametros.csv", index=False)

    # ---- melhor config de cada tipo -> várias sementes -> TESTE ----
    melhor_global = df_busca.loc[df_busca.val_loss.idxmin()]
    res, por_classe, hist_tipo, f1_tipo, cm_best = [], [], {}, {}, None
    for tipo in grade["tipo"]:
        b = df_busca[df_busca.tipo == tipo].loc[lambda x: x.val_loss.idxmin()]
        cfg = dict(tipo=tipo, hidden=int(b.hidden), camadas=int(b.camadas), lr=float(b.lr))
        print(f"\n=== Modelo final {tipo}: {cfg} (val_loss={b.val_loss:.4f}) ; {n_sem} semente(s) ===")
        f1s = []
        for s in range(n_sem):
            r = treinar(cfg, d, SEED + s, max_ep, PACIENCIA)
            pred = logits_de(r["modelo"], d["te"]["X"]).argmax(1).cpu().numpy()
            m, pc, cm = metricas(d["te"]["y_np"], pred)
            res.append(dict(modelo=tipo, semente=SEED + s, val_loss=r["val_loss"], val_acc=r["val_acc"], **m))
            f1s.append(pc["f1"])
            print(f"  semente {SEED + s}: acurácia={m['acuracia']:.4f} F1-macro={m['f1_macro']:.4f} "
                  f"(época {r['ep_melhor']}/{r['ep_rodadas']})")
            if s == 0:
                torch.save(dict(estado=r["modelo"].state_dict(), config=cfg), f"{PASTA}/modelo_{tipo}.pt")
                hist_tipo[tipo] = (r["hist"], cfg, r["ep_melhor"])
                if tipo == melhor_global.tipo:
                    cm_best = (cm, f"{tipo}, h={cfg['hidden']}, c={cfg['camadas']}")
        f1_tipo[tipo] = np.mean(f1s, axis=0)
        for c, nome in enumerate(CLASSES):
            por_classe.append(dict(modelo=tipo, classe=nome, f1=f1_tipo[tipo][c]))
        pc_ult = metricas(d["te"]["y_np"], pred)[1]  # precisão/revocação da última semente (referência)
        for c in range(len(CLASSES)):
            por_classe[-len(CLASSES) + c].update(precisao=pc_ult["precisao"][c], revocacao=pc_ult["revocacao"][c])

    # baseline: sempre prever a classe mais frequente do treino
    maj = int(np.argmax(prop_classes))
    m_base, _, _ = metricas(d["te"]["y_np"], np.full(len(d["te"]["y_np"]), maj))
    res.append(dict(modelo=f"Baseline (sempre '{CLASSES[maj]}')", semente=SEED, val_loss=np.nan, val_acc=np.nan, **m_base))

    df_res = pd.DataFrame(res)
    resumo = df_res.groupby("modelo", sort=False)[["acuracia", "f1_macro"]].agg(["mean", "std"])
    resumo.columns = [f"{a}_{b}" for a, b in resumo.columns]
    resumo.to_csv(f"{PASTA}/metricas_teste.csv")
    df_res.to_csv(f"{PASTA}/metricas_teste_por_semente.csv", index=False)
    pd.DataFrame(por_classe).to_csv(f"{PASTA}/metricas_por_classe.csv", index=False)

    figuras(d, df_busca, df_efeito, hist_tipo, cm_best[0], cm_best[1], f1_tipo)

    # ---- resumo no terminal ----
    print("\n===== EFEITO MÉDIO DE CADA HIPERPARÂMETRO (validação) =====")
    print(df_efeito.round(4).to_string(index=False))
    print("\n===== TESTE (média ± dp entre sementes) =====")
    print(resumo.round(4).to_string())
    print("\n===== F1 POR CLASSE (média entre sementes) =====")
    print(pd.DataFrame(f1_tipo, index=CLASSES).round(3).to_string())
    print(f"\nMatriz de confusão ({cm_best[1]}):\n{cm_best[0]}")
    print(f"\nArquivos salvos em: {os.path.abspath(PASTA)}")