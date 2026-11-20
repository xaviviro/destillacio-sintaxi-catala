# infer_mlx.py — inferència per lots amb mlx-lm (alumne o professor) sobre un .jsonl amb camp "frase"
import argparse, json, sys, time, importlib, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import render, extreu_json, SISTEMA
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
G = importlib.import_module("mlx_lm.generate")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--curt", type=int, default=1); ap.add_argument("--thinking", type=int, default=0)
ap.add_argument("--max_tokens", type=int, default=768); ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--camp_prompt", default=None)
a = ap.parse_args()
model, tok = load(a.model)
rows = [json.loads(l) for l in open(a.inp)]
fets = {json.loads(l)["frase"] for l in open(a.out)} if os.path.exists(a.out) else set()
pend = [r for r in rows if r["frase"] not in fets]
print(f"{len(rows)} files, {len(pend)} pendents", flush=True)
samp = make_sampler(temp=a.temperature, top_p=0.95) if a.temperature > 0 else None
t0, n = time.time(), 0
with open(a.out, "a") as fo:
    for i in range(0, len(pend), a.batch):
        lot = pend[i:i + a.batch]
        if a.camp_prompt:
            txt = [tok.apply_chat_template([{"role": "user", "content": r[a.camp_prompt]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=bool(a.thinking)) for r in lot]
        else:
            txt = [render(tok, r["frase"], bool(a.thinking), bool(a.curt)) for r in lot]
        toks = [tok.encode(t, add_special_tokens=False) for t in txt]
        kw = {"sampler": samp} if samp else {}
        res = G.batch_generate(model, tok, toks, max_tokens=a.max_tokens, **kw)
        for r, t in zip(lot, res.texts):
            out = dict(r); out["sortida"] = t
            if not a.camp_prompt:
                out["analisi"] = extreu_json(t.split("</think>")[-1])
            fo.write(json.dumps(out, ensure_ascii=False) + "\n"); n += 1
        fo.flush()
        el = time.time() - t0
        print(f"{i + len(lot)}/{len(pend)} · {el/n:.2f} s/oració · {el/60:.1f} min", flush=True)
