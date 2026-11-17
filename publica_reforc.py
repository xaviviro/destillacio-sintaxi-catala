# publica_reforc.py — publica a HF el mini 20k + reforç: dataset, model i GGUF (fitxes en català)
import json, os, re, shutil
USUARI = "xaviviro"; GH = "https://github.com/xaviviro/destillacio-sintaxi-catala"
MIDA = "20k-reforc"
DS = f"{USUARI}/sintaxi-catala-destillada-{MIDA}"; MD = f"{USUARI}/Qwen3.5-0.8B-sintaxi-catala-{MIDA}"
MD20 = f"{USUARI}/Qwen3.5-0.8B-sintaxi-catala-20k"; DS20 = f"{USUARI}/sintaxi-catala-destillada-20k"


def L(p): return [json.loads(l) for l in open(p, encoding="utf-8")]


def taula(m):  # m = {nom: {"ref":..,"test":..,"eval250":..,"crv":..,"imp":..}}
    s = "| model | referència manual (30) | test (300) | eval250 (250 a mà) | eval250 · CRV | eval250 · impersonals |\n|---|---|---|---|---|---|\n"
    return s + "\n".join(f"| {n} | {v['ref']:.1f} | {v['test']:.1f} | **{v['eval250']:.1f}** | {v['crv']:.1f} | {v['imp']:.1f} |" for n, v in m.items())


def card_dataset(n20, nref, n_test, comptes):
    return f"""---
language:
- ca
license: cc-by-4.0
task_categories:
- text-generation
pretty_name: Sintaxi catalana destil·lada — 20k + reforç
tags:
- catalan
- syntax
- distillation
- synthetic
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: test
    path: test.jsonl
- config_name: nomes_reforc
  data_files: reforc.jsonl
---

# sintaxi-catala-destillada-20k-reforc

Anàlisis sintàctiques d'oracions simples en català (JSON de gramàtica escolar) **destil·lades de Qwen3.8-27B**.
Conté el dataset [{DS20}](https://huggingface.co/datasets/{DS20}) ({n20} exemples) més una **segona ronda de {nref} exemples de reforç**.
El reforç va dirigit als punts febles que va trobar l'avaluació independent [eval250](https://huggingface.co/datasets/xaviviro/sintaxi-catala-eval250).
Projecte: [{GH}]({GH}).

## Com s'ha fet el reforç

1. El professor (Qwen3.8-27B) escriu 4.757 oracions noves demanant-li sobretot complement de règim verbal (35 %), impersonals (25 %), pronoms febles (30 %) i complement predicatiu (10 %).
   Per al CRV se li dona una pista («pensar en, parlar de, confiar en, dependre de…»).
2. El professor les analitza **a cegues**, amb raonament, sense saber quina estructura s'havia demanat.
3. Filtre determinista (`valida()`) + coincidència entre l'anàlisi a cegues i l'estructura demanada: en queden **{nref}** (22 %).
4. Es descarta qualsevol oració que ja fos als datasets, al test de 300, a la referència manual o a l'eval250.

Funcions al reforç: {comptes}.

## Fitxers

- `train.jsonl`: {n20 + nref} exemples (20k + reforç, barrejats).
- `reforc.jsonl`: només els {nref} exemples de reforç (configuració `nomes_reforc`).
- `test.jsonl`: el mateix test de {n_test} que la resta de mides.

El format és el mateix que a [{DS20}](https://huggingface.co/datasets/{DS20}).

## Limitacions

- Les etiquetes són del professor, filtrades automàticament: no són anotació humana.
- El reforç s'ha dirigit als grups on l'eval250 fallava. Cap oració de l'eval250 no és aquí, però la millora en aquests grups no és una mesura del tot independent.

## Llicència i autoria

CC BY 4.0. Xavier Vinaixa Roselló ([xaviviro](https://huggingface.co/xaviviro)), Sorensen AI, Barcelona, Catalunya.
"""


def card_model(m, us):
    return f"""---
language:
- ca
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
datasets:
- {DS}
pipeline_tag: text-generation
tags:
- catalan
- syntax
- distillation
---

# Qwen3.5-0.8B · analitzador sintàctic del català (20k + reforç)

Qwen3.5-0.8B destil·lat de **Qwen3.8-27B** per fer l'anàlisi sintàctica escolar d'oracions simples en català.
Escrius **només l'oració** i respon amb el JSON. És el [mini 20k](https://huggingface.co/{MD20}) reentrenat des del model base amb el mateix
dataset més **1.033 exemples de reforç** dirigits als seus punts febles: complement de règim verbal, impersonals, pronoms febles i CPRED.
Dataset: [{DS}](https://huggingface.co/datasets/{DS}). Versió 4 bits per a LM Studio: [{MD}-GGUF](https://huggingface.co/{MD}-GGUF).
Projecte: [{GH}]({GH}).

## Resultats (mesurats, anàlisi exacta %)

{taula(m)}

El reforç puja l'eval250 de 73,2 a 80,8 i el CRV de 6,7 a 80,0. També millora la referència manual de 30 oracions, que no s'ha fet servir per triar el reforç (73,3 → 83,3).
Alguns grups baixen una mica (predicat nominal, subjecte el·líptic), però són 2 oracions de 20–25.

{us}

## Entrenament

SFT complet amb TRL, 3 èpoques, lr 2e-5 (cosinus), batch 16, bf16, pèrdua només sobre la resposta, prompt curt (només l'oració, sense raonament).
Mateixa configuració que les altres mides.

## Autoria

Xavier Vinaixa Roselló ([xaviviro](https://huggingface.co/xaviviro)), Sorensen AI, Barcelona, Catalunya. Apache 2.0.
"""


def us_del_20k():
    from huggingface_hub import hf_hub_download
    md = open(hf_hub_download(MD20, "README.md", force_download=True), encoding="utf-8").read()
    m = re.search(r"\n## Ús.*?(?=\n## )", md, re.S)
    return m.group(0).strip().replace("sintaxi-catala-20k", f"sintaxi-catala-{MIDA}") if m else ""


def publica(token, BASE, metr, comptes):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    # dataset
    d = "/content/up_ds_reforc"; os.makedirs(d, exist_ok=True)
    t20, rf, te = L(f"{BASE}/train.jsonl"), L(f"{BASE}/reforc/train_reforc.jsonl"), L(f"{BASE}/test.jsonl")
    import random
    tr = t20 + rf; random.Random(1).shuffle(tr)
    for nom, rows in [("train", tr), ("reforc", rf), ("test", te)]:
        with open(f"{d}/{nom}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    open(f"{d}/README.md", "w", encoding="utf-8").write(card_dataset(len(t20), len(rf), len(te), comptes))
    api.create_repo(DS, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(folder_path=d, repo_id=DS, repo_type="dataset", commit_message="Dataset 20k + reforç")
    # model (bf16, sense training_args)
    src = f"{BASE}/alumne_destillat_20k_reforc"; up = "/content/up_model_reforc"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if os.path.exists(up): shutil.rmtree(up)
    AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16).save_pretrained(up)
    AutoTokenizer.from_pretrained(src).save_pretrained(up)
    for f in os.listdir(src):
        if f.endswith((".json", ".jinja", ".txt")) and not os.path.exists(f"{up}/{f}"):
            shutil.copy(f"{src}/{f}", up)
    open(f"{up}/README.md", "w", encoding="utf-8").write(card_model(metr, us_del_20k()))
    api.create_repo(MD, exist_ok=True, private=False)
    api.upload_folder(folder_path=up, repo_id=MD, commit_message="Model 20k + reforç")
    return DS, MD
