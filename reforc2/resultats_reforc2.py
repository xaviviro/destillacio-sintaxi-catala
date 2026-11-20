# resultats_reforc2.py — afegeix el mini 20k + reforç 1 + reforç 2 als resultats i regenera els gràfics de l'eval250
import json, os, sys
AQUI = os.path.dirname(os.path.abspath(__file__)); ARREL = os.path.join(AQUI, "..")
sys.path.insert(0, ARREL)
from common import avalua
import eval250_analisi as E

pred = [json.loads(l) for l in open(os.path.join(AQUI, "pred_20k_reforc2_eval250.jsonl"))]
res = json.load(open(os.path.join(ARREL, "resultats", "resultats_eval250.json")))
grups = {}
for g in dict.fromkeys(r["grup"] for r in pred):
    sub = [r for r in pred if r["grup"] == g]
    grups[g] = dict(avalua([r["analisi"] for r in sub], [r["ref"] for r in sub], [r["frase"] for r in sub]), n=len(sub))
res["20k_reforc2"] = {"global": avalua([r["analisi"] for r in pred], [r["ref"] for r in pred], [r["frase"] for r in pred]),
                      "grups": grups, "ms_per_frase": "—", "tokens_prompt_mitjana": 22.1}
# l'ordre de les claus decideix l'ordre de les barres
res = {k: res[k] for k, _ in E.NOMS if k in res}
json.dump(res, open(os.path.join(ARREL, "resultats", "resultats_eval250.json"), "w"), ensure_ascii=False, indent=1)
E.grafic_global(res, os.path.join(ARREL, "resultats", "eval250_global.png"))
E.grafic_grups(res, os.path.join(ARREL, "resultats", "eval250_grups.png"))
R = json.load(open(os.path.join(ARREL, "resultats", "resultats.json")))
c = json.load(open(os.path.join(AQUI, "resultats_colab.json")))
R["20k_reforc2"] = {"ref_exacte": c["ref30"]["exacte"], "test_exacte": c["test300"]["exacte"], "eval250_exacte": c["eval250"]["exacte"],
                    "n_reforc2": 459, "n_train": 8141}
json.dump(R, open(os.path.join(ARREL, "resultats", "resultats.json"), "w"), ensure_ascii=False, indent=1)
print(E.taula_md(res)); print(); print(E.taula_grups_md(res))
