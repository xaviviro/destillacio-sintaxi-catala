# avalua.py — mètriques globals i per grup d'un fitxer de prediccions (frase, ref, analisi[, grup])
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import avalua
def carrega(p): return [json.loads(l) for l in open(p)]
def informe(p, grups=True):
    rs = carrega(p)
    g = avalua([x["analisi"] for x in rs], [x["ref"] for x in rs], [x["frase"] for x in rs])
    out = {"global": g}
    if grups and "grup" in rs[0]:
        out["grups"] = {k: round(avalua([x["analisi"] for x in rs if x["grup"] == k], [x["ref"] for x in rs if x["grup"] == k],
                                  [x["frase"] for x in rs if x["grup"] == k])["exacte"], 1) for k in dict.fromkeys(x["grup"] for x in rs)}
    return out
if __name__ == "__main__":
    for p in sys.argv[1:]: print(p, json.dumps(informe(p), ensure_ascii=False))
