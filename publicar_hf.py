# publicar_hf.py — publica els datasets i els models destil·lats a Hugging Face (xaviviro)
# Ús a Colab:  exec(open('publicar_hf.py').read());  publica('4k', METRIQUES, ...)
import json, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import HfApi

BASE = "/content/drive/MyDrive/destillacio_sintaxi"
GITHUB = "https://github.com/xaviviro/destillacio-sintaxi-catala"
USUARI = "xaviviro"
ALUMNE = "Qwen/Qwen3.5-0.8B"
PROFESSOR = "Qwen/Qwen3.8-27B"


def noms(mida):
    return (f"{USUARI}/sintaxi-catala-destillada-{mida}", f"{USUARI}/Qwen3.5-0.8B-sintaxi-catala-{mida}")


def grafic(metriques, path, titol):
    """metriques: {nom_model: {'exacte':..,'F1_complements':..,'subjecte':..,'nucli_verbal':..}}"""
    mets = ["exacte", "F1_complements", "subjecte", "nucli_verbal"]
    etiq = ["anàlisi\nexacta", "F1\ncomplements", "subjecte", "nucli\nverbal"]
    cols = ["#E45756", "#F58518", "#54A24B", "#4C78A8", "#B279A2"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    n = len(metriques); w = 0.8 / n; x = np.arange(len(mets))
    for i, (nom, m) in enumerate(metriques.items()):
        b = ax.bar(x + (i - (n - 1) / 2) * w, [m[k] for k in mets], w, label=nom, color=cols[i % len(cols)])
        ax.bar_label(b, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, etiq); ax.set_ylim(0, 112); ax.set_ylabel("%")
    ax.set_title(titol, loc="left", fontsize=13, fontweight="bold")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(n, 3), fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def taula_md(metriques, cols=("exacte", "F1_complements", "subjecte", "nucli_verbal", "analisi_valida", "tokens_prompt", "ms_frase")):
    cap = {"exacte": "anàlisi exacta", "F1_complements": "F1 complements", "subjecte": "subjecte", "nucli_verbal": "nucli verbal",
           "analisi_valida": "sortida vàlida", "tokens_prompt": "tokens de prompt", "ms_frase": "ms/frase*"}
    cols = [c for c in cols if any(c in m for m in metriques.values())]
    out = "| model | " + " | ".join(cap[c] for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for nom, m in metriques.items():
        fmt = lambda c: ("—" if c not in m or m[c] is None else (f"{m[c]:.0f}" if c in ("tokens_prompt", "ms_frase") else f"{m[c]:.1f} %"))
        out += f"| {nom} | " + " | ".join(fmt(c) for c in cols) + " |\n"
    return out


def exemples_md(exemples):
    """exemples: llista de (frase, resultat_base, resultat_destillat)"""
    out = "| oració | Qwen3.5-0.8B base (amb prompt) | destil·lat (només l'oració) |\n|---|---|---|\n"
    for f, b, d in exemples:
        out += f"| {f} | {b} | {d} |\n"
    return out


COMU_TASCA = """**La tasca.** Donada una oració simple en català, retornar-ne l'anàlisi sintàctica segons la gramàtica escolar, en JSON: subjecte (i si és el·líptic), nucli verbal, tipus de predicat, veu, impersonalitat i complements (CD, CI, ATRIBUT, CPRED, CRV, CAG i CC de temps, lloc, manera, quantitat, companyia, instrument, causa i finalitat).

```json
{"subjecte": {"text": "L'àvia", "nucli": "àvia", "eliptic": false},
 "predicat": {"tipus": "verbal", "nucli": "serveix",
              "complements": [{"funcio": "CD", "text": "la sopa"}, {"funcio": "CI", "text": "als néts"}]},
 "veu": "activa", "impersonal": false}
```"""


def card_dataset(mida, info):
    ds, model = noms(mida)
    return f"""---
language:
- ca
license: cc-by-4.0
task_categories:
- text-generation
- token-classification
tags:
- catalan
- syntax
- distillation
- synthetic
pretty_name: Anàlisi sintàctica del català (destil·lació, {mida})
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: test
    path: test.jsonl
- config_name: referencia_manual
  data_files:
  - split: test
    path: referencia_manual.jsonl
- config_name: brut
  data_files:
  - split: train
    path: anotacio_professor*.jsonl
---

# Anàlisi sintàctica del català — dataset destil·lat ({mida})

Dataset sintètic d'oracions simples en català amb la seva anàlisi sintàctica escolar en JSON, generat i anotat per **[{PROFESSOR}](https://huggingface.co/{PROFESSOR})** i filtrat automàticament. És el dataset amb què s'ha entrenat **[{model}](https://huggingface.co/{model})**.

Forma part d'una demo de destil·lació (d'un model gran a un de molt petit) per a una xerrada sobre models petits. Codi i notebook: **[{GITHUB}]({GITHUB})**.

{COMU_TASCA}

## Com s'ha fet

1. **Generació.** El professor escriu oracions sobre {info['temes']} temes quotidians, cadascuna amb una **estructura sintàctica demanada** a l'atzar (tipus de subjecte, tipus de predicat, veu i complements). Total generat: **{info['generades']} oracions**.
2. **Anàlisi a cegues.** El mateix professor, en mode raonament (*thinking*), analitza cada oració **sense saber l'estructura que s'havia demanat**.
3. **Filtre automàtic.** Només es queden les anàlisis que (a) superen comprovacions deterministes — tots els fragments són literals de l'oració i no se solapen, el nucli del subjecte és dins del subjecte, predicat nominal ⇔ atribut, CAG ⇒ passiva… — i (b) **coincideixen amb l'estructura demanada**. Passen **{info['passen']} de {info['generades']} ({info['pct']:.0f} %)**.
4. **Partició.** {info['test']} oracions de test (fixes per a totes les mides del dataset) i **{info['train']} d'entrenament**.

### Motius de descart

{info['motius_md']}

### Distribució de funcions (train)

{info['funcions_md']}

## Fitxers

| fitxer | contingut |
|---|---|
| `train.jsonl`, `test.jsonl` | oracions filtrades: `frase`, `analisi` (JSON), `estructura` |
| `referencia_manual.jsonl` | 30 oracions anotades a mà (camp `ref`), per avaluar |
| `frases*.jsonl` | totes les oracions generades, amb l'estructura demanada i el tema |
| `anotacio_professor*.jsonl` | totes les anàlisis del professor, **sense filtrar** (inclou la sortida crua amb el raonament) |

## Limitacions — llegiu-ho abans de fer-lo servir

- **És sintètic i no està revisat per humans.** El filtre és estructural: comprova coherència, no correcció lingüística. Hi queden oracions poc naturals, castellanismes i faltes (p. ex. a la mostra crua: «Plourien les ganes de descanso pels col·legues dimarts»).
- **Les etiquetes són del professor.** El test mesura fins a quin punt s'imita el professor, no la correcció absoluta. Per a això hi ha la referència manual de 30 oracions (també sense revisió externa).
- **Està desequilibrat**: algunes funcions (CRV, CC de quantitat, CC de companyia) hi són molt poc representades, perquè el filtre les descarta més sovint.
- Només **oracions simples**; ni subordinades ni coordinades.

## Llicència i autoria

CC BY 4.0. Generat amb {PROFESSOR} (Apache 2.0). Autor: Xavier Vinaixa Roselló ([xaviviro](https://huggingface.co/xaviviro)).
"""


def card_model(mida, metr_ref, metr_test, exemples, info, altres_mides):
    ds, model = noms(mida)
    altres = ("\n\n" + "\n".join(f"- Versió amb el dataset de {m}: [{noms(m)[1]}](https://huggingface.co/{noms(m)[1]})" for m in altres_mides)) if altres_mides else ""
    return f"""---
language:
- ca
license: apache-2.0
base_model: {ALUMNE}
datasets:
- {ds}
pipeline_tag: text-generation
library_name: transformers
tags:
- catalan
- syntax
- distillation
- small-model
---

# Qwen3.5-0.8B · analitzador sintàctic del català ({mida})

Un **model de 0,8B** que, donada **només una oració** en català, respon directament amb la seva **anàlisi sintàctica en JSON**. Sense instruccions, sense exemples al prompt: la tasca i el format s'han destil·lat als pesos.

És el resultat de destil·lar **[{PROFESSOR}](https://huggingface.co/{PROFESSOR})** (27B, en mode raonament) a **[{ALUMNE}](https://huggingface.co/{ALUMNE})** amb el dataset **[{ds}](https://huggingface.co/datasets/{ds})** ({info['train']} exemples d'entrenament). Demo per a una xerrada sobre models petits; codi i notebook a **[{GITHUB}]({GITHUB})**.{altres}

{COMU_TASCA}

## Resultats (mesurats)

### Referència manual — 30 oracions anotades a mà

![resultats](resultats_referencia.png)

{taula_md(metr_ref)}
\\* ms/frase: temps mitjà per oració processant les 30 en lot amb vLLM en una RTX PRO 6000 (96 GB); no és la latència d'una crida aïllada. El professor raona abans de respondre; el petit base i el professor reben un prompt amb instruccions i un exemple (~580–615 tokens), el destil·lat només l'oració.

### Test — {info['test']} oracions (etiquetes del professor, filtrades)

{taula_md(metr_test, cols=("exacte", "F1_complements", "subjecte", "nucli_verbal", "analisi_valida", "tokens_prompt"))}
Aquest test mesura **com imita el professor**, no la correcció absoluta.

### Exemples: verbs incoatius (oracions fora del dataset)

{exemples_md(exemples)}

## Ús

```python
from transformers import AutoTokenizer, AutoModelForImageTextToText
import json, torch
m = "{model}"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForImageTextToText.from_pretrained(m, dtype=torch.bfloat16).eval()
text = tok.apply_chat_template([{{"role": "user", "content": "L'àvia serveix la sopa als néts."}}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=300, do_sample=False)
print(json.loads(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)))
```

Amb vLLM: `vllm serve {model}` i envieu l'oració com a únic missatge d'usuari (amb `enable_thinking: false`).

## Entrenament

- Fine-tuning complet (no LoRA) de {ALUMNE}, 3 èpoques, lr 2e-5 (cosinus, 20 passos d'escalfament), batch 16, bf16, pèrdua només sobre la resposta (TRL `SFTTrainer`).
- Entrada: només l'oració (missatge d'usuari). Sortida: el JSON del professor, sense raonament.
- Destil·lació per seqüència (SFT sobre les sortides filtrades del professor).

## Limitacions

- Només oracions simples. Falla sobretot en les funcions poc representades al dataset (p. ex. confon el **complement predicatiu** amb el CD: «Els avis envelleixen *tranquils* al poble»).
- De vegades **no copia literalment** el text de l'oració (p. ex. «obeeixen» → «oboeyen»); comproveu que els fragments siguin literals abans de fer-los servir.
- Hereta els errors del professor i el biaix del dataset sintètic. No s'ha avaluat amb anotadors humans externs.

## Autoria

Xavier Vinaixa Roselló ([xaviviro](https://huggingface.co/xaviviro)), Sorensen AI, Barcelona, Catalunya. Llicència Apache 2.0 (com el model base).
"""


def publica(mida, model_dir, dataset_dir, metr_ref, metr_test, exemples, info, altres_mides=(), token=None):
    api = HfApi(token=token)
    ds, model = noms(mida)
    # dataset
    api.create_repo(ds, repo_type="dataset", private=False, exist_ok=True)
    open(f"{dataset_dir}/README.md", "w", encoding="utf-8").write(card_dataset(mida, info))
    api.upload_folder(repo_id=ds, repo_type="dataset", folder_path=dataset_dir,
                      commit_message=f"Dataset de sintaxi catalana destil·lat ({mida})")
    # model: bf16 + fitxers de configuració
    up = f"/content/up_model_{mida}"
    if not os.path.exists(f"{up}/config.json"):
        import torch
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        mod = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16)
        mod.save_pretrained(up, safe_serialization=True)
        AutoTokenizer.from_pretrained(model_dir).save_pretrained(up)
        for f in os.listdir(model_dir):
            if f.endswith((".json", ".jinja", ".txt")) and not os.path.exists(f"{up}/{f}") and "safetensors" not in f:
                shutil.copy(f"{model_dir}/{f}", up)
        del mod
    grafic(metr_ref, f"{up}/resultats_referencia.png", "Referència manual · 30 oracions")
    open(f"{up}/README.md", "w", encoding="utf-8").write(card_model(mida, metr_ref, metr_test, exemples, info, altres_mides))
    api.create_repo(model, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=model, repo_type="model", folder_path=up, ignore_patterns=["training_args.bin"],
                      commit_message=f"Qwen3.5-0.8B destil·lat per a l'anàlisi sintàctica del català ({mida})")
    return f"https://huggingface.co/datasets/{ds}", f"https://huggingface.co/{model}"
