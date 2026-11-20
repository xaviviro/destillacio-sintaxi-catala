---
language:
- ca
license: cc-by-4.0
task_categories:
- text-generation
pretty_name: Sintaxi catalana destil·lada — 20k + reforç 1 + reforç 2
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
- config_name: nomes_reforc2
  data_files: reforc2.jsonl
---

# sintaxi-catala-destillada-20k-reforc2

Anàlisis sintàctiques d'oracions simples en català (JSON de gramàtica escolar). És el dataset
[xaviviro/sintaxi-catala-destillada-20k-reforc](https://huggingface.co/datasets/xaviviro/sintaxi-catala-destillada-20k-reforc)
(6.649 exemples destil·lats de Qwen3.8-27B + 1.033 de reforç) més un **segon reforç de 459 oracions escrites i anotades a mà**.
Projecte: [github.com/xaviviro/destillacio-sintaxi-catala](https://github.com/xaviviro/destillacio-sintaxi-catala).

## Com s'ha fet el segon reforç

1. S'analitzen els 48 errors del mini 20k + reforç a l'eval250 i s'agrupen per **categories d'error** (no per oracions):
   CC de quantitat i de temps confosos amb CPRED o lloc («molt», «sovint», «a les set»), CC de causa amb «de»
   («de fred»), CPRED fusionat amb el CD («la porta oberta»), impersonals amb «fer» («Fa fred»: nucli «Fa» + CD),
   verbs psicològics («M'agraden les maduixes»: subjecte postverbal, CI feble i reduplicat), CI amb «a» + persona
   confós amb CC de lloc, CRV, passiva perifràstica en futur i predicat nominal amb «estar».
2. Per a cada categoria s'escriuen directament oracions noves amb la seva anàlisi, seguint les convencions de l'eval250.
   No les ha generat el professor. Les ha revisat el corrector Escalf (LanguageTool, variant central).
3. Totes passen el mateix filtre determinista (`valida()`) que les dades destil·lades.
4. Es descarta qualsevol oració que coincideixi amb el train, el test de 300, la referència manual de 30 o l'eval250,
   i es reescriuen les que s'assemblaven massa a una oració de l'eval250 (Jaccard de paraules ≥ 0,6).

| categoria | oracions |
|---|---|
| complements circumstancials | 100 |
| complement predicatiu | 55 |
| CD + CI | 40 |
| pronoms febles i verbs psicològics | 64 |
| impersonals | 41 |
| predicat nominal i passiva | 42 |
| complement de règim verbal | 40 |
| còpia literal (noms propis, verbs incoatius) | 30 |
| barreja | 47 |

Funcions al reforç 2: CD 201, CC_LLOC 114, CC_TEMPS 112, CI 106, CPRED 62, CRV 37, CC_QUANTITAT 36, ATRIBUT 26,
CC_CAUSA 21, CC_INSTRUMENT 16, CC_MANERA 9, CC_COMPANYIA 9, CAG 4, CC_FINALITAT 3 · impersonals 43.

**Avís sobre l'eval250.** Les categories del reforç surten dels errors a l'eval250 (com al primer reforç), de manera que
l'eval250 ja no és del tot cec per a aquest model. Per això també s'informa de la referència de 30 i del test de 300.

## Fitxers

- `train.jsonl`: 8.141 exemples (20k + reforç 1 + reforç 2, barrejats). El camp `origen` diu de quina part ve cada exemple.
- `reforc2.jsonl`: només els 459 exemples del segon reforç (configuració `nomes_reforc2`), amb el camp `categoria`.
- `test.jsonl`: el mateix test de 300 que la resta de mides.

El format és el mateix que a [xaviviro/sintaxi-catala-destillada-20k](https://huggingface.co/datasets/xaviviro/sintaxi-catala-destillada-20k).

## Resultats

Model entrenat amb aquest `train.jsonl`: [xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc2](https://huggingface.co/xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc2)
(MLX 4 bits: [...-20k-reforc2-MLX-4bit](https://huggingface.co/xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc2-MLX-4bit)).

Temperatura 0, entrada = només l'oració. Mateix codi de mètriques que la resta de models (`common.avalua`).

| | eval250 · anàlisi exacta | F1 complements | F1 funcions | subjecte | nucli verbal | JSON vàlid | referència manual (30) | test (300) |
|---|---|---|---|---|---|---|---|---|
| mini 20k + reforç (ronda anterior) | 80,8 | 87,4 | 90,1 | 97,2 | 97,2 | 100 | 83,3 | 93,0 |
| **mini 20k + 2 reforços, bf16** | **94,4** | 97,4 | 98,0 | 99,6 | 97,2 | 100 | **96,7** | 94,0 |
| mini 20k + 2 reforços, MLX 4 bits | 94,8 | 97,0 | 98,1 | 99,6 | 98,4 | 100 | 96,7 | 93,7 |
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
| CPRED (15) | 60 | 80 | 80 |
| CRV (15) | 80 | 100 | 100 |
| CC (50) | 78 | 92 | 94 |
| verbs incoatius (25) | 80 | 100 | 100 |
| subjecte el·líptic (20) | 80 | 95 | 90 |

El bf16 s'ha avaluat amb `transformers` (Colab) i amb `mlx-lm` (Mac): tots dos donen el 94,4 % a l'eval250.

**Avís.** Les categories dels dos reforços es van triar mirant els errors a l'eval250. Cap oració de l'eval250 no s'ha fet servir
per entrenar, però per a aquest model l'eval250 ja no és una mesura independent: el 94,4 % s'ha de llegir com a límit superior.
La referència manual de 30 oracions, que no es va fer servir per triar res, passa del 83,3 al 96,7 %.


## Llicència i cita

CC BY 4.0. Xavier Vinaixa Roselló i Maria Sevilla Paris (2026), *Mil exemples ben triats: dades dirigides contra volum
en la destil·lació de l'anàlisi sintàctica del català*.
