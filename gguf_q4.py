# gguf_q4.py — converteix un model destil·lat a GGUF Q4_K_M (per a LM Studio / llama.cpp), el valida i el publica
import json, os, re, shutil, subprocess, time, urllib.request

LLAMA = "/content/llama.cpp"


def sh(cmd, **kw):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-4000:]); raise RuntimeError(cmd)
    return p.stdout


def prepara_llama():
    if not os.path.exists(f"{LLAMA}/build/bin/llama-quantize"):
        sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA}")
        sh(f"pip install -q -r {LLAMA}/requirements/requirements-convert_hf_to_gguf.txt")
        sh("pip install -q -U transformers")  # els requirements de llama.cpp el rebaixen i no coneix Qwen3_5Tokenizer
        sh(f"cmake -S {LLAMA} -B {LLAMA}/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release")
        sh(f"cmake --build {LLAMA}/build --target llama-quantize llama-server -j 8")


def plantilla_sense_thinking(src_dir):
    """El model s'ha entrenat sense raonament: la plantilla de xat el desactiva sempre."""
    p = f"{src_dir}/chat_template.jinja"
    t = open(p, encoding="utf-8").read()
    if not t.startswith("{%- set enable_thinking = false %}"):
        open(p, "w", encoding="utf-8").write("{%- set enable_thinking = false %}\n" + t)
    tc = f"{src_dir}/tokenizer_config.json"
    cfg = json.load(open(tc, encoding="utf-8"))
    if "chat_template" in cfg and isinstance(cfg["chat_template"], str) and not cfg["chat_template"].startswith("{%- set enable_thinking"):
        cfg["chat_template"] = "{%- set enable_thinking = false %}\n" + cfg["chat_template"]
        json.dump(cfg, open(tc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def converteix(model_dir, mida):
    src = f"/content/gguf_src_{mida}"
    if os.path.exists(src): shutil.rmtree(src)
    shutil.copytree(model_dir, src)
    plantilla_sense_thinking(src)
    out = f"/content/gguf_{mida}"; os.makedirs(out, exist_ok=True)
    f16 = f"{out}/sintaxi-catala-{mida}-bf16.gguf"
    q4 = f"{out}/Qwen3.5-0.8B-sintaxi-catala-{mida}-Q4_K_M.gguf"
    sh(f"python {LLAMA}/convert_hf_to_gguf.py {src} --outtype bf16 --outfile {f16}")
    sh(f"{LLAMA}/build/bin/llama-quantize {f16} {q4} Q4_K_M")
    return f16, q4


def extreu(t):
    t = t.rsplit("</think>", 1)[-1]; s, e = t.find("{"), t.rfind("}")
    try: return json.loads(t[s:e + 1])
    except Exception: return None


def valida_q4(q4, referencia_bf16, frases_extra=()):
    """Compara les sortides del GGUF Q4 amb les del model bf16 (vLLM) a les mateixes oracions."""
    srv = subprocess.Popen(f"{LLAMA}/build/bin/llama-server -m {q4} -ngl 99 --port 8089 -c 2048 --jinja",
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=1); break
        except Exception: time.sleep(1)

    def crida(frase):
        body = json.dumps({"messages": [{"role": "user", "content": frase}], "temperature": 0, "max_tokens": 400}).encode()
        req = urllib.request.Request("http://127.0.0.1:8089/v1/chat/completions", body, {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]

    try:
        refs = [json.loads(l) for l in open(referencia_bf16, encoding="utf-8")]
        iguals = valids = 0; t0 = time.time(); mostres = []
        for r in refs:
            txt = crida(r["frase"]); a = extreu(txt)
            valids += a is not None; iguals += a == r["analisi"]
            mostres.append((r["frase"], a))
        dt = (time.time() - t0) / len(refs)
        extra = [(f, extreu(crida(f))) for f in frases_extra]
    finally:
        import signal
        os.killpg(os.getpgid(srv.pid), signal.SIGTERM)  # el shell i el llama-server
    return {"n": len(refs), "json_valid": 100 * valids / len(refs), "identic_a_bf16": 100 * iguals / len(refs),
            "s_per_frase": dt}, mostres, extra


def card_gguf(mida, val, mida_mb):
    model = f"xaviviro/Qwen3.5-0.8B-sintaxi-catala-{mida}"
    return f"""---
language:
- ca
license: apache-2.0
base_model: {model}
pipeline_tag: text-generation
tags:
- catalan
- syntax
- gguf
- llama.cpp
- lm-studio
---

# Qwen3.5-0.8B · analitzador sintàctic del català ({mida}) — GGUF Q4_K_M

Versió **quantitzada a 4 bits (Q4_K_M, {mida_mb:.0f} MB)** de **[{model}](https://huggingface.co/{model})**, per fer-la córrer en local amb **LM Studio** (Mac, Windows, Linux) o **llama.cpp**.

Escriu **només una oració** en català i el model respon directament amb la seva anàlisi sintàctica en JSON. La plantilla de xat inclosa **desactiva sempre el raonament** (*thinking*), tal com es va entrenar.

Codi, notebook i explicació: **[github.com/xaviviro/destillacio-sintaxi-catala](https://github.com/xaviviro/destillacio-sintaxi-catala)**.

## LM Studio

1. Cerca `xaviviro/Qwen3.5-0.8B-sintaxi-catala-{mida}-GGUF` a la pestanya de descàrregues (o `lms get xaviviro/Qwen3.5-0.8B-sintaxi-catala-{mida}-GGUF`).
2. Obre un xat nou **sense prompt de sistema**, temperatura **0**.
3. Escriu una oració: `L'àvia serveix la sopa als néts.`

## llama.cpp

```bash
llama-server -hf xaviviro/Qwen3.5-0.8B-sintaxi-catala-{mida}-GGUF --jinja
```

## Validació (mesurada)

Hem comparat aquest Q4_K_M amb el model original en bf16 sobre les **{val['n']} oracions de la referència manual** (temperatura 0):

| | resultat |
|---|---|
| sortida JSON vàlida | {val['json_valid']:.0f} % |
| anàlisi **idèntica** a la del model bf16 | {val['identic_a_bf16']:.0f} % |

Aquesta taula mesura **quant s'assembla el Q4 a l'original**, no si l'anàlisi és correcta: per a la qualitat de l'anàlisi, vegeu els resultats a la fitxa del model original.

## Autoria

Xavier Vinaixa Roselló ([xaviviro](https://huggingface.co/xaviviro)), Sorensen AI, Barcelona, Catalunya. Apache 2.0.
"""


def publica_gguf(mida, q4, val, token):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    repo = f"xaviviro/Qwen3.5-0.8B-sintaxi-catala-{mida}-GGUF"
    d = f"/content/up_gguf_{mida}"; os.makedirs(d, exist_ok=True)
    shutil.copy(q4, d)
    open(f"{d}/README.md", "w", encoding="utf-8").write(card_gguf(mida, val, os.path.getsize(q4) / 1e6))
    api.create_repo(repo, private=False, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=d, commit_message=f"GGUF Q4_K_M del model {mida}")
    return f"https://huggingface.co/{repo}"
