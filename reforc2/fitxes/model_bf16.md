---
language:
- ca
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
datasets:
- xaviviro/sintaxi-catala-destillada-20k-reforc2
- xaviviro/sintaxi-catala-eval250
pipeline_tag: text-generation
tags:
- catalan
- syntax
- distillation
---

# Qwen3.5-0.8B · analitzador sintàctic del català (20k + 2 reforços)

El millor dels minis: **94,4 %** d'anàlisis exactes a l'eval250 (250 oracions noves anotades a mà) i **96,7 %** a la referència manual,
amb 0,8B paràmetres. Per fer-lo córrer en un Mac: **[versió MLX 4 bits (0,44 GB)](https://huggingface.co/xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc2-MLX-4bit)**.

Escriu **només una oració** en català i el model respon directament amb la seva anàlisi sintàctica escolar en JSON
(subjecte, predicat, complements, veu i si és impersonal). Sense instruccions, sense prompt de sistema i sense raonament.

```text
Entrada:  A la Núria li encanta la música.
Sortida:  {"subjecte": {"text": "la música", "nucli": "música", "eliptic": false},
           "predicat": {"tipus": "verbal", "nucli": "encanta",
                        "complements": [{"funcio": "CI", "text": "A la Núria"}, {"funcio": "CI", "text": "li"}]},
           "veu": "activa", "impersonal": false}
```

Codi, notebook i explicació: **[github.com/xaviviro/destillacio-sintaxi-catala](https://github.com/xaviviro/destillacio-sintaxi-catala)**.

## Ús (transformers)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc2")
model = AutoModelForCausalLM.from_pretrained("xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc2", dtype="bfloat16")
p = tok.apply_chat_template([{"role": "user", "content": "L'àvia serveix la sopa als nets."}],
                            add_generation_prompt=True, tokenize=False, enable_thinking=False)
x = tok(p, return_tensors="pt", add_special_tokens=False)
print(tok.decode(model.generate(**x, max_new_tokens=400, do_sample=False)[0][x["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Avaluació (mesurada)

Temperatura 0, entrada = només l'oració. Mateix codi de mètriques que la resta de models (`common.avalua`).

| | eval250 · anàlisi exacta | F1 complements | F1 funcions | subjecte | nucli verbal | JSON vàlid | referència manual (30) | test (300) |
|---|---|---|---|---|---|---|---|---|
| mini 20k + reforç (ronda anterior) | 80,8 | 87,4 | 90,1 | 97,2 | 97,2 | 100 | 83,3 | 93,0 |
| **mini 20k + 2 reforços, bf16** | **94,4** | 97,4 | 98,0 | 99,6 | 97,2 | 100 | **96,7** | 94,0 |
| mini 20k + 2 reforços, MLX 4 bits | 94,4 | 96,7 | 97,8 | 99,6 | 98,4 | 100 | 96,7 | 93,7 |
| professor Qwen3.8-27B (raonant) | 91,2 | 95,2 | 95,5 | 98,8 | 98,0 | 99,6 | 93,3 | — |

Anàlisi exacta (%) per tipus d'oració a l'eval250:

| tipus d'oració (n) | 20k + reforç | 20k + 2 reforços, bf16 | MLX 4 bits |
|---|---|---|---|
| CD (30) | 100 | 93 | 93 |
| CD+CI (20) | 85 | 100 | 100 |
| pronoms febles (15) | 60 | 100 | 100 |
| predicat nominal (25) | 88 | 96 | 96 |
| veu passiva (20) | 95 | 100 | 100 |
| impersonal (15) | 60 | 80 | 87 |
| CPRED (15) | 60 | 80 | 73 |
| CRV (15) | 80 | 100 | 100 |
| CC (50) | 78 | 92 | 94 |
| verbs incoatius (25) | 80 | 100 | 100 |
| subjecte el·líptic (20) | 80 | 95 | 90 |

El bf16 s'ha avaluat amb `transformers` (Colab) i amb `mlx-lm` (Mac): tots dos donen el 94,4 % a l'eval250.

**Avís.** Les categories dels dos reforços es van triar mirant els errors a l'eval250. Cap oració de l'eval250 no s'ha fet servir
per entrenar, però per a aquest model l'eval250 ja no és una mesura independent: el 94,4 % s'ha de llegir com a límit superior.
La referència manual de 30 oracions, que no es va fer servir per triar res, passa del 83,3 al 96,7 %.

Prediccions i mètriques: `resultats/pred_eval250.jsonl`, `resultats/resultats.json`.

## Com s'ha fet

1. **Destil·lació**: Qwen3.8-27B escriu oracions amb una estructura sintàctica demanada i les analitza a cegues; només es conserven
   les anàlisis que passen un validador determinista i coincideixen amb l'estructura demanada (20k: 6.649 exemples).
2. **Reforç 1**: 1.033 exemples més del professor, centrats en CRV, impersonals, pronoms febles i CPRED.
3. **Reforç 2**: 459 oracions escrites i anotades directament (sense el professor) per a les categories d'error que encara quedaven:
   CC de quantitat, temps i causa, CPRED amb CD, impersonals amb «fer», verbs psicològics, CI amb «a» + persona, CRV, passiva en futur
   i predicat nominal amb «estar». Revisades amb un corrector i validades amb el mateix filtre.
   Dataset: [xaviviro/sintaxi-catala-destillada-20k-reforc2](https://huggingface.co/datasets/xaviviro/sintaxi-catala-destillada-20k-reforc2).
4. **Entrenament** des de `Qwen/Qwen3.5-0.8B` amb els 8.141 exemples: ajust complet (TRL `SFTTrainer`), 3 èpoques, lr 2e-5 cosinus,
   lots de 16, bf16, pèrdua només sobre la resposta. Entrada = només l'oració. Una NVIDIA L4, 37 minuts.

## Autoria

Xavier Vinaixa Roselló ([xaviviro](https://huggingface.co/xaviviro)), Sorensen AI, i Maria Sevilla Paris, Universitat de Barcelona. Apache 2.0.

## Citació

```bibtex
@misc{vinaixa2026sintaximodel20kreforc2,
  title        = {Qwen3.5-0.8B-sintaxi-catala-20k-reforc2: analitzador sintàctic del català},
  author       = {Vinaixa Roselló, Xavier and Sevilla Paris, Maria},
  year         = {2026},
  institution  = {Sorensen AI i Universitat de Barcelona},
  note         = {ORCID: 0009-0005-2769-9215}
}
```

Informe: Vinaixa Roselló, X. i Sevilla Paris, M. (2026). *Mil exemples ben triats: dades dirigides contra volum en la destil·lació de l'anàlisi sintàctica del català*.
