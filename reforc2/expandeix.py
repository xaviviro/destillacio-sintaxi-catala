# expandeix.py — converteix les oracions de reforç escrites a mà (notació compacta) a JSON i les valida.
# Línia:  oració | subjecte | nucli verbal | FUNCIÓ=text; FUNCIÓ=text
#   subjecte: «el» (el·líptic), «imp» (impersonal) o «text>nucli»
#   nucli: «nom:» davant = predicat nominal; «pas:» davant = veu passiva
import json, sys, os, glob
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import valida, norm, assigna_spans, estructura, REFERENCIA

def parse(l, fitxer):
    parts = [p.strip() for p in l.split("|")]
    if len(parts) != 4: raise ValueError(f"{fitxer}: {l}")
    fr, s, v, cs = parts
    tipus, veu = "verbal", "activa"
    if v.startswith("nom:"): tipus, v = "nominal", v[4:]
    if v.startswith("pas:"): veu, v = "passiva", v[4:]
    imp = s == "imp"
    subj = {"text": None, "nucli": None, "eliptic": s == "el"} if s in ("el", "imp") else \
           {"text": s.split(">")[0].strip(), "nucli": s.split(">")[1].strip(), "eliptic": False}
    comps = []
    for c in [c.strip() for c in cs.split(";") if c.strip()]:
        f, t = c.split("=", 1); comps.append({"funcio": f.strip(), "text": t.strip()})
    # ordre dels complements = ordre d'aparició a l'oració
    sp = assigna_spans(fr, [v] + [c["text"] for c in comps])
    if sp: comps = [c for _, c in sorted(zip(sp[1:], comps), key=lambda z: z[0][0])]
    a = {"subjecte": subj, "predicat": {"tipus": tipus, "nucli": v, "complements": comps}, "veu": veu, "impersonal": imp}
    return {"frase": fr, "analisi": a}

# exclusions: eval250, referència de 30 i el train/test del Hub (20k + reforç 1)
from huggingface_hub import hf_hub_download
AQUI = os.path.dirname(os.path.abspath(__file__))
excl = {norm(json.loads(l)["frase"]) for l in open(os.path.join(AQUI, "..", "resultats", "eval250.jsonl"))}
for f in ("train.jsonl", "test.jsonl"):
    p = hf_hub_download("xaviviro/sintaxi-catala-destillada-20k-reforc", f, repo_type="dataset")
    excl |= {norm(json.loads(l)["frase"]) for l in open(p)}
excl |= {norm(f) for f, _ in REFERENCIA}
rows, vistes, mal, cat = [], set(), [], Counter()
for f in sorted(glob.glob(os.path.join(AQUI, "dades", "*.txt"))):
    for l in open(f):
        l = l.strip()
        if not l or l.startswith("#"): continue
        r = parse(l, f); ok, m = valida(r["frase"], r["analisi"])
        n = norm(r["frase"])
        if not ok: mal.append((os.path.basename(f), m, r["frase"])); continue
        if n in excl: mal.append((os.path.basename(f), "ja és a l'eval250 o al train", r["frase"])); continue
        if n in vistes: mal.append((os.path.basename(f), "repetida", r["frase"])); continue
        vistes.add(n); r["estructura"] = estructura(r["analisi"]); r["categoria"] = os.path.basename(f)[:-4]
        rows.append(r); cat[r["categoria"]] += 1
with open(os.path.join(AQUI, "train_reforc2.jsonl"), "w") as fo:
    for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "vàlides ·", dict(cat))
for m in mal: print("  ✗", *m)
