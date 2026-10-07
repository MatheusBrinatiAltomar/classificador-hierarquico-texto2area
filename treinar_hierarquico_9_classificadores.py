"""
Treina nove classificadores especializados em XGBoost + CUDA usando a saída do
modelo publicado do texto2area, um para cada Grande Área da CAPES.

Pipeline:
    texto cru
        -> texto2area (9 grandes áreas)
        -> XGBoost especializado correspondente à grande área (roteamento)

Para treino, o corpus `dados/corpus_td_lemas.parquet` é usado. A coluna
`lemmas_ext` já está no formato pré-processado consumido pelo vetorizador.

Features do segundo estágio, derivadas da saída do texto2area:
    1. 9 margens de `decision_function`;
    2. grande área prevista, em one-hot (9 atributos);
    3. até N termos decisivos, em hashing esparso, com prefixo de ranking.

Balanceamento:
    - undersampling aleatório exato antes do split;
    - todas as classes de cada especialista ficam com a quantidade da menor
      classe da própria base;
    - o split posterior é estratificado.

Instalação:
    pip install -U xgboost tqdm pandas pyarrow scipy scikit-learn joblib

Exemplos:
    python reproduzir/treinar_xgboost_9_areas.py --amostra 10000
    python reproduzir/treinar_xgboost_9_areas.py

Depois do treino, testar texto cru:
    python reproduzir/treinar_xgboost_9_areas.py --sem-treinamento \
        --texto "Título. Resumo do trabalho acadêmico..."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit("Instale XGBoost com: pip install -U xgboost") from exc


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CORPUS = REPO / "dados" / "corpus_td_lemas.parquet"
T2A_DATA = REPO / "texto2area" / "data"
T2A_VECTORIZER = T2A_DATA / "vetorizador.joblib"
T2A_MODEL = T2A_DATA / "modelo.joblib"
OUT_DIR = HERE / "modelo_treinado" / "especialistas_9_areas"

SEED = 42


# ---------------------------------------------------------------------------
# Taxonomia do experimento (49 Áreas de Avaliação da CAPES)
# ---------------------------------------------------------------------------

CONFIG_AREAS = {
    "CIÊNCIAS AGRÁRIAS": [
        "CIÊNCIA DE ALIMENTOS",
        "CIÊNCIAS AGRÁRIAS I",
        "MEDICINA VETERINÁRIA",
        "ZOOTECNIA / RECURSOS PESQUEIROS",
    ],
    "CIÊNCIAS BIOLÓGICAS": [
        "BIODIVERSIDADE",
        "CIÊNCIAS BIOLÓGICAS I",
        "CIÊNCIAS BIOLÓGICAS II",
        "CIÊNCIAS BIOLÓGICAS III",
    ],
    "CIÊNCIAS DA SAÚDE": [
        "EDUCAÇÃO FÍSICA",
        "ENFERMAGEM",
        "FARMÁCIA",
        "MEDICINA I",
        "MEDICINA II",
        "MEDICINA III",
        "NUTRIÇÃO",
        "ODONTOLOGIA",
        "SAÚDE COLETIVA",
    ],
    "CIÊNCIAS EXATAS E DA TERRA": [
        "ASTRONOMIA / FÍSICA",
        "CIÊNCIA DA COMPUTAÇÃO",
        "GEOCIÊNCIAS",
        "MATEMÁTICA / PROBABILIDADE E ESTATÍSTICA",
        "QUÍMICA",
    ],
    "CIÊNCIAS HUMANAS": [
        "ANTROPOLOGIA / ARQUEOLOGIA",
        "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS",
        "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "EDUCAÇÃO",
        "FILOSOFIA",
        "GEOGRAFIA",
        "HISTÓRIA",
        "PSICOLOGIA",
        "SOCIOLOGIA",
    ],
    "CIÊNCIAS SOCIAIS APLICADAS": [
        "ADMINISTRAÇÃO PÚBLICA E DE EMPRESAS, CIÊNCIAS CONTÁBEIS E TURISMO",
        "ARQUITETURA, URBANISMO E DESIGN",
        "COMUNICAÇÃO E INFORMAÇÃO",
        "DIREITO",
        "ECONOMIA",
        "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA",
        "SERVIÇO SOCIAL",
    ],
    "ENGENHARIAS": [
        "ENGENHARIAS I",
        "ENGENHARIAS II",
        "ENGENHARIAS III",
        "ENGENHARIAS IV",
    ],
    "LINGUÍSTICA, LETRAS E ARTES": [
        "ARTES",
        "LINGUÍSTICA E LITERATURA",
    ],
    "MULTIDISCIPLINAR": [
        "BIOTECNOLOGIA",
        "CIÊNCIAS AMBIENTAIS",
        "ENSINO",
        "INTERDISCIPLINAR",
        "MATERIAIS",
    ],
}

SLUGS = {
    "CIÊNCIAS AGRÁRIAS": "ciencias_agrarias",
    "CIÊNCIAS BIOLÓGICAS": "ciencias_biologicas",
    "CIÊNCIAS DA SAÚDE": "ciencias_da_saude",
    "CIÊNCIAS EXATAS E DA TERRA": "ciencias_exatas_e_da_terra",
    "CIÊNCIAS HUMANAS": "ciencias_humanas",
    "CIÊNCIAS SOCIAIS APLICADAS": "ciencias_sociais_aplicadas",
    "ENGENHARIAS": "engenharias",
    "LINGUÍSTICA, LETRAS E ARTES": "linguistica_letras_artes",
    "MULTIDISCIPLINAR": "multidisciplinar",
}

VALID_PAIRS = {
    (ga, classe)
    for ga, classes in CONFIG_AREAS.items()
    for classe in classes
}

# Configuração padrão pensando em uma GPU com 6 GB.
DEFAULT_BATCH = 25_000
DEFAULT_TOP_TERMS = 5
DEFAULT_HASH_FEATURES = 2_048
DEFAULT_N_ESTIMATORS = 500
DEFAULT_MAX_DEPTH = 6
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MIN_CHILD_WEIGHT = 5
DEFAULT_SUBSAMPLE = 0.80
DEFAULT_COLSAMPLE = 0.80
DEFAULT_MAX_BIN = 256
DEFAULT_EARLY_STOPPING = 50
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def normalizar_rotulo(valor: object) -> str:
    """Normaliza espaços/Unicode e leva o rótulo para caixa alta."""
    if pd.isna(valor):
        return ""
    texto = str(valor).replace("\u00a0", " ")
    texto = unicodedata.normalize("NFKC", texto)
    return " ".join(texto.strip().split()).upper()


def mapear_area_especialista(valor: object) -> str:
    """
    Converte os rótulos históricos encontrados no corpus nas classes atuais.
    """
    area = normalizar_rotulo(valor)

    mapa = {
        # LLA
        "ARTES / MÚSICA": "ARTES",
        "LINGUISTICA E LITERATURA": "LINGUÍSTICA E LITERATURA",
        "LETRAS / LINGUÍSTICA": "LINGUÍSTICA E LITERATURA",
        "LETRAS / LINGUISTICA": "LINGUÍSTICA E LITERATURA",

        # Humanas
        "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS": "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS",
        "CIENCIA POLITICA E RELACOES INTERNACIONAIS": "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS",
        "CIÊNCIAS DA RELIGIÃO E TEOLOGIA": "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "CIENCIAS DA RELIGIAO E TEOLOGIA": "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "TEOLOGIA": "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "FILOSOFIA/TEOLOGIA:SUBCOMISSÃO TEOLOGIA": "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "EDUCAÇÃO": "EDUCAÇÃO",
        "EDUCACAO": "EDUCAÇÃO",
        "FILOSOFIA/TEOLOGIA:SUBCOMISSÃO FILOSOFIA": "FILOSOFIA",
        "HISTÓRIA": "HISTÓRIA",
        "HISTORIA": "HISTÓRIA",

        # Exatas
        "ASTRONOMIA / FISICA": "ASTRONOMIA / FÍSICA",
        "ASTRONOMIA/FÍSICA": "ASTRONOMIA / FÍSICA",
        "COMPUTAÇÃO / CIÊNCIA DA COMPUTAÇÃO": "CIÊNCIA DA COMPUTAÇÃO",
        "COMPUTAÇÃO": "CIÊNCIA DA COMPUTAÇÃO",
        "CIENCIA DA COMPUTACAO": "CIÊNCIA DA COMPUTAÇÃO",
        "GEOCIENCIAS": "GEOCIÊNCIAS",
        "MATEMATICA / PROBABILIDADE E ESTATISTICA": "MATEMÁTICA / PROBABILIDADE E ESTATÍSTICA",
        "MATEMÁTICA / PROBABILIDADE ESTATÍSTICA": "MATEMÁTICA / PROBABILIDADE E ESTATÍSTICA",
        "QUIMICA": "QUÍMICA",

        # Biológicas
        "CIENCIAS BIOLOGICAS I": "CIÊNCIAS BIOLÓGICAS I",
        "CIENCIAS BIOLOGICAS II": "CIÊNCIAS BIOLÓGICAS II",
        "CIENCIAS BIOLOGICAS III": "CIÊNCIAS BIOLÓGICAS III",

        # Saúde
        "EDUCACAO FISICA": "EDUCAÇÃO FÍSICA",
        "FARMACIA": "FARMÁCIA",
        "NUTRICAO": "NUTRIÇÃO",
        "ODONTOLOGIA": "ODONTOLOGIA",
        "SAUDE COLETIVA": "SAÚDE COLETIVA",

        # Agrárias
        "CIENCIAS AGRARIAS I": "CIÊNCIAS AGRÁRIAS I",
        "CIENCIA DE ALIMENTOS": "CIÊNCIA DE ALIMENTOS",
        "MEDICINA VETERINARIA": "MEDICINA VETERINÁRIA",
        "ZOOTECNIA / RECURSOS PESQUEIROS": "ZOOTECNIA / RECURSOS PESQUEIROS",
        "ZOOTECNIA/RECURSOS PESQUEIROS": "ZOOTECNIA / RECURSOS PESQUEIROS",

        # Sociais Aplicadas
        "ADMINISTRACAO PUBLICA E DE EMPRESAS, CIENCIAS CONTABEIS E TURISMO": "ADMINISTRAÇÃO PÚBLICA E DE EMPRESAS, CIÊNCIAS CONTÁBEIS E TURISMO",
        "ADMINISTRAÇÃO, CIÊNCIAS CONTÁBEIS E TURISMO": "ADMINISTRAÇÃO PÚBLICA E DE EMPRESAS, CIÊNCIAS CONTÁBEIS E TURISMO",
        "COMUNICACAO E INFORMACAO": "COMUNICAÇÃO E INFORMAÇÃO",
        "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA": "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA",
        "PLANEJAMENTO URBANO E REGIONAL/DEMOGRAFIA": "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA",
        "SERVICO SOCIAL": "SERVIÇO SOCIAL",

        # Multidisciplinar
        "CIENCIAS AMBIENTAIS": "CIÊNCIAS AMBIENTAIS",
        "CIÊNCIA DOS MATERIAIS": "MATERIAIS",
    }
    return mapa.get(area, area)


def titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def carregar_corpus(batch_size: int, amostra: int | None) -> pd.DataFrame:
    """Lê as colunas necessárias e filtra as Grandes Áreas em streaming."""
    if not CORPUS.exists():
        raise FileNotFoundError(
            f"Corpus não encontrado: {CORPUS}\n"
            "Baixe o `corpus_td_lemas.parquet` do release do repositório."
        )

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(CORPUS)
    total = pf.metadata.num_rows
    colunas = ["id_producao", "grande_area", "area_avaliacao", "lemmas_ext"]

    partes: list[pd.DataFrame] = []
    mantidos = 0
    grandes_validas = set(CONFIG_AREAS.keys())

    with tqdm(
        total=total,
        desc="Lendo corpus",
        unit=" docs",
        dynamic_ncols=True,
    ) as bar:
        for tabela in pf.iter_batches(batch_size=batch_size, columns=colunas):
            bloco = tabela.to_pandas()

            grande = bloco["grande_area"].map(normalizar_rotulo)
            bloco = bloco.loc[grande.isin(grandes_validas)].copy()
            if bloco.empty:
                bar.update(len(tabela))
                continue

            bloco["grande_area"] = grande.loc[bloco.index]
            bloco["area_classe"] = bloco["area_avaliacao"].map(mapear_area_especialista)

            bloco = bloco.dropna(subset=["lemmas_ext", "area_classe"])
            bloco = bloco[bloco["lemmas_ext"].str.strip() != ""]

            mask = pd.Series([
                (ga, ac) in VALID_PAIRS
                for ga, ac in zip(bloco["grande_area"], bloco["area_classe"])
            ], index=bloco.index)
            
            bloco = bloco.loc[mask]

            if not bloco.empty:
                partes.append(bloco)
                mantidos += len(bloco)

            bar.update(len(tabela))
            if amostra is not None and mantidos >= amostra:
                break

    if not partes:
        raise RuntimeError("Nenhum documento das áreas validadas foi encontrado.")

    df = pd.concat(partes, ignore_index=True)
    if amostra is not None and len(df) > amostra:
        df = df.sample(n=amostra, random_state=SEED).reset_index(drop=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Texto2area: reproduz as saídas publicadas
# ---------------------------------------------------------------------------

def carregar_texto2area():
    """Carrega exatamente os artefatos publicados do texto2area."""
    if not T2A_VECTORIZER.exists():
        raise FileNotFoundError(f"Não encontrei: {T2A_VECTORIZER}")
    if not T2A_MODEL.exists():
        raise FileNotFoundError(f"Não encontrei: {T2A_MODEL}")

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    vec = joblib.load(T2A_VECTORIZER)
    clf = joblib.load(T2A_MODEL)
    return vec, clf


def extrair_termos_decisivos(
    X: sparse.csr_matrix,
    clf,
    nomes_features: np.ndarray,
    pred_idx: np.ndarray,
    topo: int,
) -> list[list[str]]:
    resultados: list[list[str]] = []
    coef = clf.coef_

    for i in range(X.shape[0]):
        ini = X.indptr[i]
        fim = X.indptr[i + 1]
        if ini == fim:
            resultados.append([])
            continue

        indices = X.indices[ini:fim]
        valores = X.data[ini:fim]
        k = int(pred_idx[i])
        contribuicao = valores * coef[k, indices]
        n = min(topo, contribuicao.size)

        if contribuicao.size > n:
            cand = np.argpartition(contribuicao, -n)[-n:]
        else:
            cand = np.arange(contribuicao.size)

        cand = cand[np.argsort(contribuicao[cand])[::-1]]
        resultados.append([str(nomes_features[indices[j]]) for j in cand])

    return resultados


def features_saida_texto2area(
    textos: pd.Series,
    vec,
    clf,
    batch_size: int,
    top_terms: int,
    hash_features: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Constrói as features finais e retorna também a grande área prevista."""
    nomes_features = np.asarray(vec.get_feature_names_out())
    classes = np.asarray(clf.classes_, dtype=object)
    chunks: list[sparse.csr_matrix] = []
    predicoes: list[np.ndarray] = []

    hasher = FeatureHasher(
        n_features=hash_features,
        input_type="string",
        alternate_sign=False,
        dtype=np.float32,
    )

    with tqdm(
        total=len(textos),
        desc="Gerando saída do texto2area",
        unit=" docs",
        dynamic_ncols=True,
    ) as bar:
        for ini in range(0, len(textos), batch_size):
            fim = min(ini + batch_size, len(textos))
            bloco = textos.iloc[ini:fim].to_numpy()

            X_base = vec.transform(bloco).tocsr()
            margens = np.asarray(clf.decision_function(X_base))
            if margens.ndim == 1:
                margens = margens.reshape(-1, 1)

            idx_pred = np.argmax(margens, axis=1).astype(np.int32)
            pred = classes[idx_pred]
            predicoes.append(pred)

            termos = extrair_termos_decisivos(
                X_base, clf, nomes_features, idx_pred, top_terms,
            )

            onehot = sparse.csr_matrix(
                (
                    np.ones(len(idx_pred), dtype=np.float32),
                    (np.arange(len(idx_pred)), idx_pred),
                ),
                shape=(len(idx_pred), len(classes)),
            )

            entradas_hash = [
                [f"r{rank}:{term}" for rank, term in enumerate(ts, 1)]
                for ts in termos
            ]
            hashed = hasher.transform(entradas_hash).tocsr()
            X_margin = sparse.csr_matrix(margens.astype(np.float32, copy=False))

            chunks.append(
                sparse.hstack([X_margin, onehot, hashed], format="csr", dtype=np.float32)
            )
            bar.update(len(bloco))

    return sparse.vstack(chunks, format="csr", dtype=np.float32), np.concatenate(predicoes)


# ---------------------------------------------------------------------------
# Callback de progresso do XGBoost
# ---------------------------------------------------------------------------

class TQDMCallback(xgb.callback.TrainingCallback):
    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc
        self.bar = None

    def before_training(self, model):
        self.bar = tqdm(
            total=self.total,
            desc=self.desc,
            unit=" árvores",
            dynamic_ncols=True,
        )
        return model

    def after_iteration(self, model, epoch: int, evals_log):
        if self.bar is None:
            return False

        self.bar.n = min(epoch + 1, self.total)
        try:
            if evals_log:
                dataset = list(evals_log.keys())[-1]
                metricas = evals_log[dataset]
                nome = list(metricas.keys())[-1]
                valor = metricas[nome][-1]
                self.bar.set_postfix(**{nome: f"{valor:.5f}"})
        except Exception:
            pass
        self.bar.refresh()
        return False

    def after_training(self, model):
        if self.bar is not None:
            self.bar.close()
        return model


# ---------------------------------------------------------------------------
# Treinamento/eval de uma cabeça
# ---------------------------------------------------------------------------

def split_estratificado(
    y: np.ndarray,
    test_size: float,
    val_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx,
        y,
        test_size=test_size + val_size,
        random_state=SEED,
        stratify=y,
    )

    frac_test = test_size / (test_size + val_size)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp,
        y_temp,
        test_size=frac_test,
        random_state=SEED,
        stratify=y_temp,
    )
    return idx_train, idx_val, idx_test


def balancear_dataset_por_undersampling(
    y: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    contagens = np.array(
        [np.sum(y == classe) for classe in range(n_classes)],
        dtype=int,
    )
    alvo = int(contagens.min())

    if alvo <= 0:
        raise ValueError(
            "Pelo menos uma classe não possui documentos e não pode ser "
            "balanceada para o mesmo tamanho."
        )

    partes = []
    registros = []

    for classe in range(n_classes):
        idx_classe = np.flatnonzero(y == classe)
        quantidade_antes = len(idx_classe)

        escolhidos = rng.choice(
            idx_classe,
            size=alvo,
            replace=False,
        )
        partes.append(escolhidos)

        registros.append({
            "classe_id": classe,
            "quantidade_antes": quantidade_antes,
            "quantidade_depois": alvo,
            "quantidade_descartada": quantidade_antes - alvo,
            "alvo_balanceamento": alvo,
        })

    idx_balanceado = np.concatenate(partes)
    rng.shuffle(idx_balanceado)

    contagens_depois = np.bincount(y[idx_balanceado], minlength=n_classes)
    if not np.all(contagens_depois == alvo):
        raise RuntimeError("Falha no balanceamento das classes.")

    diagnostico = pd.DataFrame(registros)
    diagnostico["quantidade_depois"] = contagens_depois
    return idx_balanceado, diagnostico

def treinar_especialista(
    nome: str,
    X: sparse.csr_matrix,
    labels: np.ndarray,
    classes: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    titulo(f"XGBOOST — {nome.upper()}")

    classes_to_int = {c: i for i, c in enumerate(classes)}
    inesperadas = sorted(set(labels) - set(classes))
    if inesperadas:
        raise ValueError(f"Classes inesperadas em {nome}: {inesperadas}")

    y_original = np.asarray(
        [classes_to_int[str(v)] for v in labels],
        dtype=np.int32,
    )

    print(f"Documentos originais: {len(y_original):,}")
    print(f"Features:             {X.shape[1]:,}")
    print(f"Classes:              {len(classes)}")
    print("Distribuição original:")
    contagem_original = pd.Series(labels).value_counts().reindex(
        classes, fill_value=0
    )
    for c, n in contagem_original.items():
        print(f"  {c}: {n:,}")

    if args.balancear_treino:
        idx_balanceado, balanceamento = balancear_dataset_por_undersampling(
            y_original, len(classes)
        )
        X = X[idx_balanceado]
        y = y_original[idx_balanceado]
    else:
        balanceamento = pd.DataFrame({
            "classe_id": np.arange(len(classes)),
            "quantidade_antes": [int(np.sum(y_original == i)) for i in range(len(classes))],
            "quantidade_depois": [int(np.sum(y_original == i)) for i in range(len(classes))],
            "quantidade_descartada": [0] * len(classes),
            "alvo_balanceamento": [np.nan] * len(classes),
        })
        y = y_original

    print("\nDistribuição após balanceamento:")
    contagem_balanceada = pd.Series(y).value_counts().reindex(
        np.arange(len(classes)), fill_value=0
    )
    for i, c in enumerate(classes):
        print(f"  {c}: {int(contagem_balanceada.iloc[i]):,}")

    if args.balancear_treino:
        alvo = int(balanceamento["alvo_balanceamento"].iloc[0])
        print(f"\nUndersampling ativado.")
        print(f"Quantidade-alvo por classe: {alvo:,} documentos")
        print(f"Total após balanceamento:   {len(y):,} documentos")

    idx_train, idx_val, idx_test = split_estratificado(
        y, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE
    )

    X_train = X[idx_train]
    X_val = X[idx_val]
    X_test = X[idx_test]
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    objetivo_binario = len(classes) == 2
    objective = "binary:logistic" if objetivo_binario else "multi:softprob"
    eval_metric = "logloss" if objetivo_binario else "mlogloss"

    params = dict(
        objective=objective,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        max_bin=args.max_bin,
        tree_method="hist",
        device="cuda",
        eval_metric=eval_metric,
        random_state=SEED,
        early_stopping_rounds=args.early_stopping,
        callbacks=[TQDMCallback(args.n_estimators, f"Treinando {nome}")],
    )
    if len(classes) > 2:
        params["num_class"] = len(classes)

    print("\nDispositivo XGBoost: CUDA")

    model = xgb.XGBClassifier(**params)
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    tempo = time.time() - t0

    print("\nAvaliando...")
    with tqdm(total=1, desc=f"Avaliando {nome}", unit=" etapa", dynamic_ncols=True) as bar:
        y_pred = model.predict(X_test).astype(np.int32)
        probs = model.predict_proba(X_test)
        bar.update(1)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nMétricas:")
    print(f"  Accuracy:          {acc:.6f}")
    print(f"  Balanced accuracy: {bal_acc:.6f}")
    print(f"  F1 macro:          {f1_macro:.6f}")
    print(f"  F1 weighted:       {f1_weighted:.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)

    balanceamento_out = balanceamento.copy()
    balanceamento_out["classe"] = [classes[int(i)] for i in balanceamento_out["classe_id"]]
    balanceamento_out.to_csv(out_dir / f"balanceamento_{nome}.csv", index=False, encoding="utf-8-sig")

    model_path = out_dir / f"modelo_{nome}.json"
    model.save_model(model_path)

    pred_df = pd.DataFrame({
        "y_true": [classes[int(v)] for v in y_test],
        "y_pred": [classes[int(v)] for v in y_pred],
        "confianca": probs.max(axis=1),
    })
    pred_df.to_csv(out_dir / f"predicoes_teste_{nome}.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        out_dir / f"matriz_confusao_{nome}.csv", encoding="utf-8-sig"
    )

    report = classification_report(
        y_test, y_pred, labels=np.arange(len(classes)),
        target_names=classes, output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(out_dir / f"classification_report_{nome}.csv", encoding="utf-8-sig")

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp_rows = [{"feature": k, "gain": float(v)} for k, v in gain.items()]
    if imp_rows:
        pd.DataFrame(imp_rows).sort_values("gain", ascending=False).to_csv(
            out_dir / f"importancia_gain_{nome}.csv", index=False, encoding="utf-8-sig"
        )

    meta = {
        "nome": nome,
        "classes": classes,
        "n_documentos": int(len(y_original)),
        "n_documentos_original": int(len(y_original)),
        "n_documentos_balanceados": int(len(y)),
        "n_treino": int(len(idx_train)),
        "n_validacao": int(len(idx_val)),
        "n_teste": int(len(idx_test)),
        "balanceamento": {
            "metodo": "undersampling_aleatorio",
            "aplicado_antes_do_split": bool(args.balancear_treino),
            "quantidade_alvo_por_classe": (
                int(balanceamento["alvo_balanceamento"].iloc[0]) if args.balancear_treino else None
            ),
        },
        "n_features": int(X.shape[1]),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_score": float(getattr(model, "best_score", np.nan)),
        "segundos_treino": round(tempo, 3),
        "params": {k: v for k, v in params.items() if k != "callbacks"},
    }
    (out_dir / f"treino_{nome}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    return {"model": model, "classes": classes, "metadata": meta}


# ---------------------------------------------------------------------------
# Inferência: texto cru -> texto2area -> especialista
# ---------------------------------------------------------------------------

def features_de_saida_unica(
    area: str,
    margens: list[tuple[str, float]],
    termos: list[str],
    classes_grandes: list[str],
    hash_features: int,
) -> sparse.csr_matrix:
    margem_dict = {str(c): float(v) for c, v in margens}
    margens_vetor = np.asarray(
        [margem_dict.get(c, 0.0) for c in classes_grandes],
        dtype=np.float32,
    ).reshape(1, -1)

    if area not in classes_grandes:
        raise ValueError(f"Grande área inesperada: {area}")
    idx = classes_grandes.index(area)

    onehot = sparse.csr_matrix(
        (np.array([1.0], dtype=np.float32), ([0], [idx])),
        shape=(1, len(classes_grandes)),
    )

    hasher = FeatureHasher(
        n_features=hash_features,
        input_type="string",
        alternate_sign=False,
        dtype=np.float32,
    )
    tokens = [[f"r{i}:{t}" for i, t in enumerate(termos, start=1)]]
    hashed = hasher.transform(tokens).tocsr()

    return sparse.hstack(
        [sparse.csr_matrix(margens_vetor), onehot, hashed],
        format="csr", dtype=np.float32,
    )


def testar_texto(
    texto: str,
    modelos: dict[str, dict],
    classes_grandes: list[str],
    hash_features: int,
) -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from texto2area import classificar

    area, margens, termos = classificar(texto, topo=DEFAULT_TOP_TERMS)

    titulo("PREDIÇÃO EM TEXTO CRU")
    print(f"Grande área prevista: {area}")
    print("Margens:")
    for c, v in margens:
        print(f"  {c}: {v:.6f}")
    print(f"Termos decisivos: {termos}")

    chave = SLUGS.get(area)
    if not chave or chave not in modelos:
        print(f"\nNenhum modelo especialista em memória para a área '{area}'.")
        return

    info = modelos[chave]
    X = features_de_saida_unica(area, margens, termos, classes_grandes, hash_features)
    pred = int(info["model"].predict(X)[0])
    prob = info["model"].predict_proba(X)[0]

    print(f"\nEspecialista acionado: {chave}")
    print(f"Área de avaliação prevista: {info['classes'][pred]}")
    print("Probabilidades:")
    for c, p in sorted(zip(info["classes"], prob), key=lambda x: -x[1]):
        print(f"  {c}: {p:.6f}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--amostra", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--top-termos", type=int, default=DEFAULT_TOP_TERMS)
    ap.add_argument("--hash-features", type=int, default=DEFAULT_HASH_FEATURES)
    ap.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    ap.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    ap.add_argument("--min-child-weight", type=int, default=DEFAULT_MIN_CHILD_WEIGHT)
    ap.add_argument("--subsample", type=float, default=DEFAULT_SUBSAMPLE)
    ap.add_argument("--colsample-bytree", type=float, default=DEFAULT_COLSAMPLE)
    ap.add_argument("--max-bin", type=int, default=DEFAULT_MAX_BIN)
    ap.add_argument("--early-stopping", type=int, default=DEFAULT_EARLY_STOPPING)
    ap.add_argument(
        "--sem-balanceamento", dest="balancear_treino", action="store_false",
        help="desativa o undersampling do treino (para comparação experimental)",
    )
    ap.set_defaults(balancear_treino=True)
    ap.add_argument("--texto", type=str, default=None)
    ap.add_argument(
        "--sem-treinamento", action="store_true",
        help="carrega os modelos já salvos e usa somente --texto",
    )
    return ap.parse_args()


def carregar_modelos_salvos() -> dict[str, dict]:
    modelos: dict[str, dict] = {}
    for ga, slug in SLUGS.items():
        classes = CONFIG_AREAS[ga]
        path = OUT_DIR / f"modelo_{slug}.json"
        if not path.exists():
            continue
        model = xgb.XGBClassifier()
        model.load_model(path)
        modelos[slug] = {"model": model, "classes": classes}
        
    if not modelos:
        raise FileNotFoundError(f"Nenhum modelo encontrado no diretório: {OUT_DIR}")
    return modelos


def main() -> None:
    args = parse_args()
    inicio_total = time.time()

    titulo("XGBOOST ESPECIALIZADO — TODAS AS 9 GRANDES ÁREAS")
    print(f"Repositório: {REPO}")
    print(f"Corpus:      {CORPUS}")
    print(f"Saída:       {OUT_DIR}")
    print("GPU:         device='cuda'")

    if args.sem_treinamento:
        if not args.texto:
            raise SystemExit("--sem-treinamento exige --texto")
        modelos = carregar_modelos_salvos()
        _vec, t2a_clf = carregar_texto2area()
        classes_grandes = [str(c) for c in t2a_clf.classes_]
        testar_texto(args.texto, modelos, classes_grandes, args.hash_features)
        return

    titulo("1/5 — CARREGAMENTO DO CORPUS")
    df = carregar_corpus(args.batch_size, args.amostra)
    print(f"Documentos selecionados: {len(df):,}")
    print("\nGrandes áreas:")
    print(df["grande_area"].value_counts().to_string())

    titulo("2/5 — GERAÇÃO DA SAÍDA DO TEXTO2AREA")
    vec, clf = carregar_texto2area()
    classes_grandes = [str(c) for c in clf.classes_]

    X_all, pred_grande = features_saida_texto2area(
        df["lemmas_ext"], vec, clf, args.batch_size, args.top_termos, args.hash_features,
    )
    df["grande_area_predita_texto2area"] = pred_grande

    titulo("3/5 — DIAGNÓSTICO DO ROTEAMENTO")
    print(pd.crosstab(df["grande_area"], df["grande_area_predita_texto2area"], margins=True).to_string())
    for grande in CONFIG_AREAS.keys():
        mask = df["grande_area"].eq(grande)
        if mask.any():
            taxa = np.mean(df.loc[mask, "grande_area_predita_texto2area"].to_numpy() == grande)
            print(f"Roteamento correto para {grande}: {taxa:.4%}")

    titulo("4/5 — TREINAMENTO DOS ESPECIALISTAS")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": SEED,
        "corpus": str(CORPUS),
        "texto2area_model": str(T2A_MODEL),
        "texto2area_vectorizer": str(T2A_VECTORIZER),
        "features": {
            "margens": len(classes_grandes),
            "one_hot_grande_area": len(classes_grandes),
            "top_termos": args.top_termos,
            "hash_features": args.hash_features,
        },
        "balanceamento": {
            "metodo": "undersampling_aleatorio",
            "antes_do_split": True,
            "ativo": bool(args.balancear_treino),
        },
        "xgboost": {
            "device": "cuda",
            "tree_method": "hist",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "max_bin": args.max_bin,
            "early_stopping_rounds": args.early_stopping,
        },
    }

    for ga, classes in CONFIG_AREAS.items():
        config[SLUGS[ga]] = {
            "grande_area": ga,
            "classes": classes,
        }

    (OUT_DIR / "configuracao_experimento.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    resumos = []
    modelos_em_memoria = {}

    for ga, classes in CONFIG_AREAS.items():
        chave = SLUGS[ga]
        mask_ga = df["grande_area"].eq(ga).to_numpy()

        if not mask_ga.any():
            print(f"\nAviso: Nenhuma amostra encontrada para {ga}. Pulando...")
            continue

        try:
            res = treinar_especialista(
                chave,
                X_all[mask_ga],
                df.loc[mask_ga, "area_classe"].to_numpy(),
                classes,
                OUT_DIR,
                args,
            )
            resumos.append({
                "especialista": ga,
                "documentos": res["metadata"]["n_documentos"],
                "documentos_balanceados": res["metadata"]["n_documentos_balanceados"],
                "treino": res["metadata"]["n_treino"],
                "accuracy": res["metadata"]["accuracy"],
                "balanced_accuracy": res["metadata"]["balanced_accuracy"],
                "f1_macro": res["metadata"]["f1_macro"],
                "f1_weighted": res["metadata"]["f1_weighted"],
            })
            modelos_em_memoria[chave] = {
                "model": res["model"],
                "classes": res["classes"]
            }
        except Exception as e:
            print(f"\n[ERRO] Falha ao treinar especialista {ga}: {e}")

    titulo("5/5 — RESUMO")
    if resumos:
        df_resumo = pd.DataFrame(resumos)
        print(df_resumo.to_string(index=False))
        df_resumo.to_csv(OUT_DIR / "resumo_experimento.csv", index=False, encoding="utf-8-sig")

    print(f"\nTempo total: {time.time() - inicio_total:,.1f} s")
    print(f"Artefatos: {OUT_DIR}")

    if args.texto:
        testar_texto(args.texto, modelos_em_memoria, classes_grandes, args.hash_features)

if __name__ == "__main__":
    main()