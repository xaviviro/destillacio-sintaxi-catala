# Destil·lació: d'un Qwen gran a un Qwen petit

Demo per a una xerrada sobre models petits: un **analitzador sintàctic del català** (oració simple → JSON amb subjecte, nucli verbal, CD, CI, atribut, CPRED, CRV, CAG, CC…) destil·lat d'un Qwen gran a un de molt petit, tot a Google Colab.

## Flux

1. **El petit sense entrenar** (Qwen3.5-0.8B) sobre 30 oracions anotades a mà → no se'n surt.
2. **El professor** (Qwen3.8-27B) escriu ~4.000 oracions amb una estructura sintàctica demanada i després les analitza *a cegues*.
3. **Filtre**: validació determinista (fragments literals, sense solapaments, coherència nominal⇔atribut, CAG⇔passiva…) + coincidència entre l'anàlisi a cegues i l'estructura demanada.
4. **Destil·lació**: SFT complet de l'alumne amb les anàlisis filtrades (sense raonament).
5. **Avaluació**: petit base vs destil·lat vs professor, sobre la referència manual i un test de 300 oracions.

## Fitxers

| fitxer | què fa |
|---|---|
| `destillacio_sintaxi_catala.ipynb` | el notebook de Colab, en format «presentació» (títols numerats, codi amagat en formularis) |
| `common.py` | prompt, validació, mètriques, visualització i les 30 oracions de referència |
| `infer.py` | inferència per lots amb vLLM (en un procés a part, per alliberar la GPU) |
| `gen_frases.py` | el professor genera oracions amb estructura demanada |
| `build.py` | regenera el notebook a partir dels `.py` |

Les dades i el model es desen a Google Drive (`MyDrive/destillacio_sintaxi`) i es reutilitzen si ja hi són.

## Resultats

**Mesurat** (RTX PRO 6000 Blackwell 96 GB, Colab Pro+) — Qwen3.5-0.8B base, 30 oracions de referència:

| JSON vàlid | anàlisi coherent | subjecte | nucli verbal | F1 complements | anàlisi exacta |
|---|---|---|---|---|---|
| 100 % | 50 % | 53 % | 63 % | 18 % | **0 %** |

**Pendent de mesurar**: professor, alumne destil·lat, temps i cost.

## Advertiments

- La referència manual (30 oracions) està anotada a mà seguint la gramàtica escolar i **encara no s'ha revisat**: es pot corregir a `common.py`.
- Al test de 300 oracions les etiquetes són del professor (filtrades): mesura com imita el professor, no la correcció absoluta.
