"""
Treina nove classificadores especializados baseados em BERT (BERTimbau) 
usando PyTorch e Hugging Face Transformers, um para cada Grande Área da CAPES.

Pipeline de Treinamento:
    texto (lemmas/cru) -> Tokenizador BERT -> Fine-Tuning do modelo BERTimbau.
    (A etapa de geração de features por hashing do texto2area foi eliminada no treino, 
     pois o BERT extrai features semânticas diretamente do texto).

Pipeline de Inferência:
    texto cru 
        -> texto2area (prevê as 9 grandes áreas para roteamento)
        -> BERT especializado acionado para prever a Área de Avaliação (49 áreas).

Instalação:
    pip install torch transformers datasets accelerate scikit-learn pandas pyarrow tqdm

Exemplos:
    python reproduzir/treinar_bert_9_areas.py --amostra 10000 --epochs 3
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
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CORPUS = REPO / "dados" / "corpus_td_lemas.parquet"
T2A_DATA = REPO / "texto2area" / "data"
T2A_VECTORIZER = T2A_DATA / "vetorizador.joblib"
T2A_MODEL = T2A_DATA / "modelo.joblib"
OUT_DIR = HERE / "modelo_treinado" / "especialistas_bert_9_areas"

SEED = 42
set_seed(SEED)

# ---------------------------------------------------------------------------
# Taxonomia (49 Áreas de Avaliação da CAPES)
# ---------------------------------------------------------------------------

CONFIG_AREAS = {
    "CIÊNCIAS AGRÁRIAS": [
        "CIÊNCIA DE ALIMENTOS", "CIÊNCIAS AGRÁRIAS I", "MEDICINA VETERINÁRIA", "ZOOTECNIA / RECURSOS PESQUEIROS",
    ],
    "CIÊNCIAS BIOLÓGICAS": [
        "BIODIVERSIDADE", "CIÊNCIAS BIOLÓGICAS I", "CIÊNCIAS BIOLÓGICAS II", "CIÊNCIAS BIOLÓGICAS III",
    ],
    "CIÊNCIAS DA SAÚDE": [
        "EDUCAÇÃO FÍSICA", "ENFERMAGEM", "FARMÁCIA", "MEDICINA I", "MEDICINA II", "MEDICINA III", "NUTRIÇÃO", "ODONTOLOGIA", "SAÚDE COLETIVA",
    ],
    "CIÊNCIAS EXATAS E DA TERRA": [
        "ASTRONOMIA / FÍSICA", "CIÊNCIA DA COMPUTAÇÃO", "GEOCIÊNCIAS", "MATEMÁTICA / PROBABILIDADE E ESTATÍSTICA", "QUÍMICA",
    ],
    "CIÊNCIAS HUMANAS": [
        "ANTROPOLOGIA / ARQUEOLOGIA", "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS", "CIÊNCIAS DA RELIGIÃO E TEOLOGIA", "EDUCAÇÃO", "FILOSOFIA", "GEOGRAFIA", "HISTÓRIA", "PSICOLOGIA", "SOCIOLOGIA",
    ],
    "CIÊNCIAS SOCIAIS APLICADAS": [
        "ADMINISTRAÇÃO PÚBLICA E DE EMPRESAS, CIÊNCIAS CONTÁBEIS E TURISMO", "ARQUITETURA, URBANISMO E DESIGN", "COMUNICAÇÃO E INFORMAÇÃO", "DIREITO", "ECONOMIA", "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA", "SERVIÇO SOCIAL",
    ],
    "ENGENHARIAS": [
        "ENGENHARIAS I", "ENGENHARIAS II", "ENGENHARIAS III", "ENGENHARIAS IV",
    ],
    "LINGUÍSTICA, LETRAS E ARTES": [
        "ARTES", "LINGUÍSTICA E LITERATURA",
    ],
    "MULTIDISCIPLINAR": [
        "BIOTECNOLOGIA", "CIÊNCIAS AMBIENTAIS", "ENSINO", "INTERDISCIPLINAR", "MATERIAIS",
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

VALID_PAIRS = {(ga, classe) for ga, classes in CONFIG_AREAS.items() for classe in classes}


# Configurações padrão para BERT
DEFAULT_BERT_MODEL = "neuralmind/bert-base-portuguese-cased"
DEFAULT_MAX_LEN = 512
DEFAULT_BATCH = 16
DEFAULT_EPOCHS = 4
DEFAULT_LR = 2e-5
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15


# ---------------------------------------------------------------------------
# Utilidades (Mantidas do XGBoost)
# ---------------------------------------------------------------------------

def normalizar_rotulo(valor: object) -> str:
    if pd.isna(valor):
        return ""
    texto = str(valor).replace("\u00a0", " ")
    texto = unicodedata.normalize("NFKC", texto)
    return " ".join(texto.strip().split()).upper()

def mapear_area_especialista(valor: object) -> str:
    area = normalizar_rotulo(valor)
    mapa = {
        "ARTES / MÚSICA": "ARTES",
        "LINGUISTICA E LITERATURA": "LINGUÍSTICA E LITERATURA",
        "LETRAS / LINGUÍSTICA": "LINGUÍSTICA E LITERATURA",
        "LETRAS / LINGUISTICA": "LINGUÍSTICA E LITERATURA",
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
        "ASTRONOMIA / FISICA": "ASTRONOMIA / FÍSICA",
        "ASTRONOMIA/FÍSICA": "ASTRONOMIA / FÍSICA",
        "COMPUTAÇÃO / CIÊNCIA DA COMPUTAÇÃO": "CIÊNCIA DA COMPUTAÇÃO",
        "COMPUTAÇÃO": "CIÊNCIA DA COMPUTAÇÃO",
        "CIENCIA DA COMPUTACAO": "CIÊNCIA DA COMPUTAÇÃO",
        "GEOCIENCIAS": "GEOCIÊNCIAS",
        "MATEMATICA / PROBABILIDADE E ESTATISTICA": "MATEMÁTICA / PROBABILIDADE E ESTATÍSTICA",
        "MATEMÁTICA / PROBABILIDADE ESTATÍSTICA": "MATEMÁTICA / PROBABILIDADE E ESTATÍSTICA",
        "QUIMICA": "QUÍMICA",
        "CIENCIAS BIOLOGICAS I": "CIÊNCIAS BIOLÓGICAS I",
        "CIENCIAS BIOLOGICAS II": "CIÊNCIAS BIOLÓGICAS II",
        "CIENCIAS BIOLOGICAS III": "CIÊNCIAS BIOLÓGICAS III",
        "EDUCACAO FISICA": "EDUCAÇÃO FÍSICA",
        "FARMACIA": "FARMÁCIA",
        "NUTRICAO": "NUTRIÇÃO",
        "ODONTOLOGIA": "ODONTOLOGIA",
        "SAUDE COLETIVA": "SAÚDE COLETIVA",
        "CIENCIAS AGRARIAS I": "CIÊNCIAS AGRÁRIAS I",
        "CIENCIA DE ALIMENTOS": "CIÊNCIA DE ALIMENTOS",
        "MEDICINA VETERINARIA": "MEDICINA VETERINÁRIA",
        "ZOOTECNIA / RECURSOS PESQUEIROS": "ZOOTECNIA / RECURSOS PESQUEIROS",
        "ZOOTECNIA/RECURSOS PESQUEIROS": "ZOOTECNIA / RECURSOS PESQUEIROS",
        "ADMINISTRACAO PUBLICA E DE EMPRESAS, CIENCIAS CONTABEIS E TURISMO": "ADMINISTRAÇÃO PÚBLICA E DE EMPRESAS, CIÊNCIAS CONTÁBEIS E TURISMO",
        "ADMINISTRAÇÃO, CIÊNCIAS CONTÁBEIS E TURISMO": "ADMINISTRAÇÃO PÚBLICA E DE EMPRESAS, CIÊNCIAS CONTÁBEIS E TURISMO",
        "COMUNICACAO E INFORMACAO": "COMUNICAÇÃO E INFORMAÇÃO",
        "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA": "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA",
        "PLANEJAMENTO URBANO E REGIONAL/DEMOGRAFIA": "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA",
        "SERVICO SOCIAL": "SERVIÇO SOCIAL",
        "CIENCIAS AMBIENTAIS": "CIÊNCIAS AMBIENTAIS",
        "CIÊNCIA DOS MATERIAIS": "MATERIAIS",
    }
    return mapa.get(area, area)

def titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# Carregamento de Dados e Balanceamento
# ---------------------------------------------------------------------------

def carregar_corpus(amostra: int | None) -> pd.DataFrame:
    import pyarrow.parquet as pq
    
    if not CORPUS.exists():
        raise FileNotFoundError(f"Corpus não encontrado: {CORPUS}")

    pf = pq.ParquetFile(CORPUS)
    colunas = ["id_producao", "grande_area", "area_avaliacao", "lemmas_ext"]
    
    partes = []
    mantidos = 0
    grandes_validas = set(CONFIG_AREAS.keys())

    with tqdm(total=pf.metadata.num_rows, desc="Lendo corpus", unit=" docs") as bar:
        for tabela in pf.iter_batches(batch_size=50000, columns=colunas):
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
        raise RuntimeError("Nenhum documento encontrado.")

    df = pd.concat(partes, ignore_index=True)
    if amostra is not None and len(df) > amostra:
        df = df.sample(n=amostra, random_state=SEED).reset_index(drop=True)
    return df.reset_index(drop=True)


def split_estratificado(y: np.ndarray, test_size: float, val_size: float) -> tuple:
    idx = np.arange(len(y))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=test_size + val_size, random_state=SEED, stratify=y
    )
    frac_test = test_size / (test_size + val_size)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=frac_test, random_state=SEED, stratify=y_temp
    )
    return idx_train, idx_val, idx_test


def balancear_dataset_por_undersampling(y: np.ndarray, n_classes: int) -> tuple:
    rng = np.random.default_rng(SEED)
    contagens = np.array([np.sum(y == classe) for classe in range(n_classes)])
    alvo = int(contagens.min())

    if alvo <= 0:
        raise ValueError("Uma classe não possui documentos. Não é possível balancear.")

    partes = []
    registros = []

    for classe in range(n_classes):
        idx_classe = np.flatnonzero(y == classe)
        escolhidos = rng.choice(idx_classe, size=alvo, replace=False)
        partes.append(escolhidos)
        registros.append({
            "classe_id": classe, "quantidade_antes": len(idx_classe),
            "quantidade_depois": alvo, "alvo_balanceamento": alvo,
        })

    idx_balanceado = np.concatenate(partes)
    rng.shuffle(idx_balanceado)
    return idx_balanceado, pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# Fine-Tuning do BERT (Especialista)
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    bal_acc = balanced_accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

def treinar_especialista_bert(
    nome: str,
    textos: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    titulo(f"BERT — {nome.upper()}")
    
    classes_to_int = {c: i for i, c in enumerate(classes)}
    y_original = np.asarray([classes_to_int[str(v)] for v in labels], dtype=np.int64)

    print(f"Documentos originais: {len(y_original):,}")
    print(f"Classes:              {len(classes)}")

    if args.balancear_treino:
        idx_balanceado, balanceamento = balancear_dataset_por_undersampling(y_original, len(classes))
        textos_bal = textos[idx_balanceado]
        y_bal = y_original[idx_balanceado]
        print(f"Total após balanceamento (Undersampling): {len(y_bal):,} documentos")
    else:
        textos_bal = textos
        y_bal = y_original
        print("Balanceamento por undersampling: DESATIVADO")
        balanceamento = pd.DataFrame()

    idx_train, idx_val, idx_test = split_estratificado(y_bal, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE)
    print(f"Treino: {len(idx_train):,} | Validação: {len(idx_val):,} | Teste: {len(idx_test):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.bert_model, 
        num_labels=len(classes),
        id2label={i: c for i, c in enumerate(classes)},
        label2id=classes_to_int
    )

    def preparar_dataset(indices):
        data = {"text": textos_bal[indices].tolist(), "labels": y_bal[indices].tolist()}
        dataset = Dataset.from_dict(data)
        return dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=args.max_len), 
            batched=True, 
            remove_columns=["text"]
        )

    ds_train = preparar_dataset(idx_train)
    ds_val = preparar_dataset(idx_val)
    ds_test = preparar_dataset(idx_test)

    model_dir = out_dir / nome
    
    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(), # Otimiza uso de VRAM se GPU suportar
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nIniciando treinamento do BERT (Fine-Tuning)...")
    t0 = time.time()
    trainer.train()
    tempo_treino = time.time() - t0

    print("\nAvaliando no conjunto de Teste...")
    test_results = trainer.predict(ds_test)
    y_pred = np.argmax(test_results.predictions, axis=-1)
    y_true = ds_test["labels"]
    
    metrics = test_results.metrics
    
    # Salvamento
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir / "modelo_final"))
    tokenizer.save_pretrained(str(model_dir / "modelo_final"))

    if not balanceamento.empty:
        balanceamento["classe"] = [classes[int(i)] for i in balanceamento["classe_id"]]
        balanceamento.to_csv(model_dir / "balanceamento.csv", index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "y_true": [classes[int(v)] for v in y_true],
        "y_pred": [classes[int(v)] for v in y_pred],
    })
    pred_df.to_csv(model_dir / "predicoes_teste.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(model_dir / "matriz_confusao.csv", encoding="utf-8-sig")

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(model_dir / "classification_report.csv", encoding="utf-8-sig")

    meta = {
        "nome": nome,
        "classes": classes,
        "n_documentos": int(len(y_original)),
        "n_treino": int(len(idx_train)),
        "n_validacao": int(len(idx_val)),
        "n_teste": int(len(idx_test)),
        "test_accuracy": float(metrics["test_accuracy"]),
        "test_balanced_accuracy": float(metrics["test_balanced_accuracy"]),
        "test_f1_macro": float(metrics["test_f1_macro"]),
        "test_f1_weighted": float(metrics["test_f1_weighted"]),
        "segundos_treino": round(tempo_treino, 3),
        "params_treino": training_args.to_dict()
    }
    
    (model_dir / "meta_treino.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Finalizado {nome}. Acurácia de teste: {metrics['test_accuracy']:.4f}")

    return {"model_path": str(model_dir / "modelo_final"), "classes": classes, "metadata": meta}


# ---------------------------------------------------------------------------
# Inferência Completa (texto2area -> especialista BERT)
# ---------------------------------------------------------------------------

def testar_texto_bert(texto: str, diretorios_modelos: dict) -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from texto2area import classificar

    area, margens, termos = classificar(texto, topo=5)
    
    titulo("PREDIÇÃO EM TEXTO CRU (INFERÊNCIA)")
    print(f"Grande área (Routing texto2area): {area}")

    chave = SLUGS.get(area)
    if not chave or chave not in diretorios_modelos:
        print(f"\nNenhum modelo especialista BERT encontrado para '{area}'.")
        return

    caminho_modelo = diretorios_modelos[chave]
    print(f"Carregando BERT especialista de: {caminho_modelo}")
    
    tokenizer = AutoTokenizer.from_pretrained(caminho_modelo)
    model = AutoModelForSequenceClassification.from_pretrained(caminho_modelo)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        pred_id = np.argmax(probs)
    
    classe_prevista = model.config.id2label[pred_id]
    
    print(f"\nEspecialista acionado: {chave}")
    print(f"Área de avaliação prevista pelo BERT: {classe_prevista}")
    print("\nProbabilidades Top 3:")
    
    classes_rankeadas = [(model.config.id2label[i], probs[i]) for i in range(len(probs))]
    classes_rankeadas.sort(key=lambda x: x[1], reverse=True)
    
    for c, p in classes_rankeadas[:3]:
        print(f"  {c}: {p:.4%}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amostra", type=int, default=None)
    ap.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    ap.add_argument("--sem-balanceamento", dest="balancear_treino", action="store_false")
    ap.set_defaults(balancear_treino=True)
    ap.add_argument("--texto", type=str, default=None)
    ap.add_argument("--sem-treinamento", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    inicio_total = time.time()

    titulo("HIERARQUIA COM BERT — TODAS AS 9 GRANDES ÁREAS")
    print(f"Modelo Base: {args.bert_model}")
    print(f"Device:      {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Modo Apenas Inferência
    if args.sem_treinamento:
        if not args.texto:
            raise SystemExit("--sem-treinamento exige argumento --texto")
        
        dir_modelos = {}
        for ga, slug in SLUGS.items():
            path_modelo = OUT_DIR / slug / "modelo_final"
            if path_modelo.exists():
                dir_modelos[slug] = str(path_modelo)
        
        if not dir_modelos:
            raise FileNotFoundError(f"Nenhum modelo BERT encontrado em {OUT_DIR}")
            
        testar_texto_bert(args.texto, dir_modelos)
        return

    # Modo Treinamento
    titulo("1/3 — CARREGAMENTO DO CORPUS")
    df = carregar_corpus(args.amostra)
    print(f"Documentos selecionados: {len(df):,}")

    titulo("2/3 — TREINAMENTO DOS ESPECIALISTAS (BERT)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    resumos = []
    
    for ga, classes in CONFIG_AREAS.items():
        chave = SLUGS[ga]
        mask_ga = df["grande_area"].eq(ga).to_numpy()

        if not mask_ga.any():
            print(f"\nAviso: Nenhuma amostra encontrada para {ga}. Pulando...")
            continue

        textos_ga = df.loc[mask_ga, "lemmas_ext"].to_numpy()
        labels_ga = df.loc[mask_ga, "area_classe"].to_numpy()

        try:
            res = treinar_especialista_bert(chave, textos_ga, labels_ga, classes, OUT_DIR, args)
            resumos.append({
                "especialista": ga,
                "n_treino": res["metadata"]["n_treino"],
                "f1_macro": res["metadata"]["test_f1_macro"],
                "accuracy": res["metadata"]["test_accuracy"],
            })
        except Exception as e:
            print(f"\n[ERRO] Falha ao treinar especialista {ga}: {e}")

    titulo("3/3 — RESUMO FINAL")
    if resumos:
        df_resumo = pd.DataFrame(resumos)
        print(df_resumo.to_string(index=False))
        df_resumo.to_csv(OUT_DIR / "resumo_experimento_bert.csv", index=False, encoding="utf-8-sig")

    print(f"\nTempo total: {time.time() - inicio_total:,.1f} s")

    if args.texto:
        # Pega os caminhos recém treinados
        dir_modelos = {slug: str(OUT_DIR / slug / "modelo_final") for slug in SLUGS.values()}
        testar_texto_bert(args.texto, dir_modelos)

if __name__ == "__main__":
    main()