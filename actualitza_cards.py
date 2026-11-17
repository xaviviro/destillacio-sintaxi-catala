# actualitza_cards.py — publica el dataset eval250 i afegeix l'avaluació independent i la citació a totes les fitxes de HF
import json, os, re, sys
sys.path.insert(0, "/content"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval250_analisi import taula_md, taula_grups_md

USUARI = "xaviviro"
EVAL_REPO = f"{USUARI}/sintaxi-catala-eval250"
GH = "https://github.com/xaviviro/destillacio-sintaxi-catala"
IMG = f"https://huggingface.co/datasets/{EVAL_REPO}/resolve/main/grafics"
MIDES = ["4k", "10k", "20k", "20k-reforc"]


def cita(ident, titol):
    return f"""## Citació

```bibtex
@misc{{vinaixa2026{ident},
  title        = {{{titol}}},
  author       = {{Vinaixa Roselló, Xavier}},
  year         = {{2026}},
  institution  = {{Sorensen AI, Barcelona}},
  note         = {{ORCID: 0009-0005-2769-9215}},
  doi          = {{Pendent}},
  url          = {{Pendent}}
}}
```
"""


TITOLS = {
    "github": ("destillaciosintaxi", "Destil·lació d'un analitzador sintàctic del català: de Qwen3.8-27B a Qwen3.5-0.8B"),
    "eval": ("sintaxieval250", "sintaxi-catala-eval250: 250 oracions catalanes anotades a mà per avaluar l'anàlisi sintàctica"),
}
for m in MIDES:
    TITOLS[f"dataset_{m}"] = (f"sintaxidataset{m.replace('-', '')}", f"sintaxi-catala-destillada-{m}: anàlisis sintàctiques del català destil·lades de Qwen3.8-27B")
    TITOLS[f"model_{m}"] = (f"sintaximodel{m.replace('-', '')}", f"Qwen3.5-0.8B-sintaxi-catala-{m}: analitzador sintàctic del català destil·lat")
    TITOLS[f"gguf_{m}"] = (f"sintaxigguf{m.replace('-', '')}", f"Qwen3.5-0.8B-sintaxi-catala-{m}-GGUF: analitzador sintàctic del català quantitzat a 4 bits")


def seccio_eval(res, mida=None):
    dest = f" El model d'aquesta fitxa és el **mini {mida.replace('-reforc', ' + reforç')}**." if mida else ""
    return f"""## Avaluació independent — eval250 (mesurada)

Test nou de **250 oracions escrites i anotades a mà**, separat de tot el procés de destil·lació: cap oració no l'ha generada el professor i
cap no coincideix amb les ~20.500 oracions generades per als datasets (comprovat). Cobreix 11 tipus d'oració, entre ells verbs incoatius,
pronoms febles, passiva, impersonals, CPRED i CRV. Dataset: [{EVAL_REPO}](https://huggingface.co/datasets/{EVAL_REPO}).{dest}

{taula_md(res)}

![eval250 global]({IMG}/eval250_global.png)

Anàlisi exacta (%) per tipus d'oració:

{taula_grups_md(res)}

![eval250 per grups]({IMG}/eval250_grups.png)

*«Mini 20k + reforç» = el 20k més 1.033 oracions de reforç dirigides a CRV, impersonals, pronoms febles i CPRED, generades i filtrades igual que la resta. Cap oració de l'eval250 no s'ha fet servir per entrenar, però el reforç es va dirigir als grups on l'eval250 fallava: en aquests grups la millora no és una mesura del tot independent. Mini destil·lats amb el prompt curt (només l'oració); mini base amb el prompt complet; professor amb raonament (thinking), temperatura 0,6.
Una anàlisi és exacta si coincideixen subjecte, nucli verbal, tipus de predicat, veu i el conjunt de complements (funció + text).*
"""


def treu_seccio(md, titol):
    return re.sub(rf"\n## {re.escape(titol)}.*?(?=\n## |\Z)", "\n", md, flags=re.S)


def insereix(md, bloc, abans_de):
    for h in abans_de:
        i = md.find(f"\n## {h}")
        if i >= 0:
            return md[:i] + "\n" + bloc + md[i:]
    return md.rstrip() + "\n\n" + bloc


def actualitza(api, repo, tipus, clau, res, mida, abans_de):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "README.md", repo_type=tipus, force_download=True)
    md = open(p, encoding="utf-8").read()
    for t in ("Avaluació independent — eval250 (mesurada)", "Citació"):
        md = treu_seccio(md, t)
    md = insereix(md, seccio_eval(res, mida), abans_de)
    md = md.rstrip() + "\n\n" + cita(*TITOLS[clau])
    api.upload_file(path_or_fileobj=md.encode("utf-8"), path_in_repo="README.md", repo_id=repo, repo_type=tipus,
                    commit_message="Avaluació independent eval250 i citació")
    return repo


def card_eval(res, n_grups):
    grups = "\n".join(f"| {g} | {n} |" for g, n in n_grups.items())
    return f"""---
language:
- ca
license: cc-by-4.0
task_categories:
- token-classification
- text-generation
pretty_name: Sintaxi catalana — eval250
size_categories:
- n<1K
tags:
- catalan
- syntax
- evaluation
- distillation
configs:
- config_name: default
  data_files: eval250.jsonl
---

# sintaxi-catala-eval250

**250 oracions simples en català, escrites i anotades a mà**, per avaluar analitzadors sintàctics en format JSON (gramàtica escolar:
subjecte, predicat, nucli verbal, complements, veu, impersonalitat). És el test independent del projecte
[destil·lació d'un analitzador sintàctic del català]({GH}).

- Cap oració no l'ha generada el model professor (Qwen3.8-27B).
- Cap oració no coincideix (després de normalitzar) amb les ~20.500 oracions generades per als datasets d'entrenament ni amb les 30 de la referència manual.
- Totes passen el validador determinista del projecte (`valida()`: els textos són fragments de l'oració, l'estructura és coherent).

## Grups

| tipus d'oració | oracions |
|---|---|
{grups}

## Format

Cada línia de `eval250.jsonl`:

```json
{{"frase": "La Júlia dona un regal al seu cosí.",
  "ref": {{"subjecte": {{"text": "La Júlia", "nucli": "Júlia", "eliptic": false}},
          "predicat": {{"tipus": "verbal", "nucli": "dona",
                       "complements": [{{"funcio": "CD", "text": "un regal"}}, {{"funcio": "CI", "text": "al seu cosí"}}]}},
          "veu": "activa", "impersonal": false}},
  "grup": "CD+CI"}}
```

Funcions: CD, CI, ATRIBUT, CPRED, CRV, CAG i CC_LLOC, CC_TEMPS, CC_MANERA, CC_QUANTITAT, CC_INSTRUMENT, CC_COMPANYIA, CC_CAUSA, CC_FINALITAT.

## Resultats (mesurats)

{taula_md(res)}

![eval250 global](grafics/eval250_global.png)

{taula_grups_md(res)}

![eval250 per grups](grafics/eval250_grups.png)

Les prediccions de cada model són a `prediccions/` i les mètriques completes a `resultats_eval250.json`.

## Limitacions

- 250 oracions: cada grup té entre 15 i 50 oracions, així que una diferència de pocs punts en un grup és soroll (1 oració de 15 = 6,7 punts).
- Les anotacions segueixen els criteris del projecte, que són els d'una gramàtica escolar. Hi ha casos que admeten més d'una anàlisi,
  com el «hi ha» impersonal, el CC vs. CRV o «de l'examen» com a CC de lloc. L'avaluació exacta penalitza qualsevol discrepància amb el criteri triat.
- Només oracions simples i sense subordinades.

## Llicència i autoria

CC BY 4.0. Xavier Vinaixa Roselló ([xaviviro](https://huggingface.co/xaviviro)), Sorensen AI, Barcelona, Catalunya.

{cita(*TITOLS['eval'])}"""


def publica_eval(api, ev_dir, res):
    from collections import Counter
    rows = [json.loads(l) for l in open(f"{ev_dir}/eval250.jsonl", encoding="utf-8")]
    api.create_repo(EVAL_REPO, repo_type="dataset", exist_ok=True, private=False)
    os.makedirs("/content/up_eval/grafics", exist_ok=True); os.makedirs("/content/up_eval/prediccions", exist_ok=True)
    import shutil
    shutil.copy(f"{ev_dir}/eval250.jsonl", "/content/up_eval/")
    shutil.copy(f"{ev_dir}/resultats_eval250.json", "/content/up_eval/")
    for g in ("eval250_global.png", "eval250_grups.png"):
        shutil.copy(f"{ev_dir}/{g}", "/content/up_eval/grafics/")
    for f in os.listdir(ev_dir):
        if f.startswith("pred_"):
            shutil.copy(f"{ev_dir}/{f}", "/content/up_eval/prediccions/")
    open("/content/up_eval/README.md", "w", encoding="utf-8").write(card_eval(res, Counter(r["grup"] for r in rows)))
    api.upload_folder(folder_path="/content/up_eval", repo_id=EVAL_REPO, repo_type="dataset",
                      commit_message="Dataset d'avaluació eval250")


def tot(token, ev_dir):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    res = json.load(open(f"{ev_dir}/resultats_eval250.json"))
    publica_eval(api, ev_dir, res)
    fets = [EVAL_REPO]
    for m in MIDES:
        fets.append(actualitza(api, f"{USUARI}/sintaxi-catala-destillada-{m}", "dataset", f"dataset_{m}", res, m,
                               ["Limitacions — llegiu-ho abans de fer-lo servir", "Limitacions", "Llicència i autoria"]))
        fets.append(actualitza(api, f"{USUARI}/Qwen3.5-0.8B-sintaxi-catala-{m}", "model", f"model_{m}", res, m,
                               ["Ús", "Entrenament", "Autoria"]))
        fets.append(actualitza(api, f"{USUARI}/Qwen3.5-0.8B-sintaxi-catala-{m}-GGUF", "model", f"gguf_{m}", res, m,
                               ["LM Studio", "Autoria"]))
    return fets
