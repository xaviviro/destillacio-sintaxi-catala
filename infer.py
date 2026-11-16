# infer.py — inferència per lots amb vLLM (s'executa en un procés a part per alliberar la GPU en acabar)
import argparse, json, sys, time
sys.path.insert(0, "/content")
from common import render, extreu_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_len", type=int, default=8192)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [render(tok, r["frase"], bool(args.thinking)) for r in rows]

    kw = dict(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_len,
              enable_prefix_caching=True, max_num_seqs=256)
    try:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as e:
        print("Reintent sense limit_mm_per_prompt:", e)
        llm = LLM(**kw)

    sp = SamplingParams(temperature=args.temperature, top_p=0.95 if args.temperature > 0 else 1.0,
                        max_tokens=args.max_tokens)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    ntok = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            txt = o.outputs[0].text
            ntok += len(o.outputs[0].token_ids)
            r = dict(r, sortida=txt, analisi=extreu_json(txt), tokens=len(o.outputs[0].token_ids))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"model": args.model, "n": len(rows), "segons": round(dt, 1), "tokens_generats": ntok,
             "tokens_per_s": round(ntok / dt, 1), "ms_per_frase": round(1000 * dt / len(rows), 1)}
    json.dump(stats, open(args.out + ".stats.json", "w"))
    print(stats)


if __name__ == "__main__":
    main()
