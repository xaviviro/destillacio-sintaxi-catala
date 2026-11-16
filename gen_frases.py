# gen_frases.py — el professor escriu oracions catalanes amb una estructura sintàctica demanada
import argparse, json, random, re, sys, time
sys.path.insert(0, "/content")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--peticions", type=int, default=500)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1714)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    TEMES = ["l'escola", "la cuina", "el mar i la pesca", "la muntanya", "la ciutat", "la feina a l'oficina",
             "la música", "la història de Catalunya", "la família", "els animals de companyia", "la tecnologia",
             "la festa major", "el mercat", "els trens i rodalies", "el bosc", "l'hospital", "la biblioteca",
             "la platja", "el teatre", "els castells", "la verema", "el pagès i l'hort", "el futbol",
             "un viatge", "el temps i el clima", "la política municipal", "un restaurant", "els jocs de taula",
             "la ciència", "una botiga de barri", "els avis", "el cinema", "un concert", "l'Ajuntament",
             "la bicicleta", "una mudança", "un museu", "el carnaval", "la neu", "una excursió escolar"]

    NOMS = {"CD": "complement directe (CD)", "CI": "complement indirecte (CI)", "ATRIBUT": "atribut",
            "CPRED": "complement predicatiu (CPRED)", "CRV": "complement de règim verbal (CRV)",
            "CAG": "complement agent (CAG)", "CC_TEMPS": "complement circumstancial de temps",
            "CC_LLOC": "complement circumstancial de lloc", "CC_MANERA": "complement circumstancial de manera",
            "CC_QUANTITAT": "complement circumstancial de quantitat",
            "CC_COMPANYIA": "complement circumstancial de companyia",
            "CC_INSTRUMENT": "complement circumstancial d'instrument",
            "CC_CAUSA": "complement circumstancial de causa", "CC_FINALITAT": "complement circumstancial de finalitat"}
    CC = ["CC_TEMPS"] * 4 + ["CC_LLOC"] * 4 + ["CC_MANERA"] * 3 + ["CC_QUANTITAT", "CC_COMPANYIA",
          "CC_INSTRUMENT", "CC_CAUSA", "CC_FINALITAT"]


    def ccs(k):
        out = []
        while len(out) < k:
            c = rng.choice(CC)
            if c not in out:
                out.append(c)
        return out


    def estructura_aleatoria():
        r = rng.random()
        subj = "explicit" if rng.random() < 0.65 else "eliptic"
        if r < 0.15:
            e = {"subjecte": subj, "tipus": "nominal", "veu": "activa", "funcions": ["ATRIBUT"] + ccs(rng.choice([0, 0, 1]))}
        elif r < 0.27:
            e = {"subjecte": "explicit", "tipus": "verbal", "veu": "passiva", "funcions": ["CAG"] + ccs(rng.choice([0, 1]))}
        elif r < 0.35:
            nucli = rng.choice([[], ["CD"]])
            e = {"subjecte": "impersonal", "tipus": "verbal", "veu": "activa", "funcions": nucli + ccs(rng.choice([1, 1, 2]))}
        else:
            nucli = rng.choice([[], ["CD"], ["CD"], ["CD"], ["CD", "CI"], ["CD", "CI"], ["CI"], ["CRV"], ["CPRED"], ["CD", "CPRED"]])
            k = rng.choice([0, 1, 1, 2]) if nucli else rng.choice([1, 2])
            e = {"subjecte": subj, "tipus": "verbal", "veu": "activa", "funcions": nucli + ccs(k)}
        e["funcions"] = sorted(set(e["funcions"]))
        e["febles"] = e["tipus"] == "verbal" and e["veu"] == "activa" and any(f in e["funcions"] for f in ("CD", "CI")) and rng.random() < 0.3
        return e


    def descriu(e):
        l = {"explicit": "- subjecte explícit", "eliptic": "- subjecte el·líptic (omès, no aparegui a l'oració)",
             "impersonal": "- oració impersonal, sense subjecte (verbs com ploure, nevar, fer fred, haver-hi...)"}[e["subjecte"]]
        l = [l]
        if e["tipus"] == "nominal":
            l.append("- predicat nominal amb verb copulatiu (ser, estar o semblar)")
        elif e["veu"] == "passiva":
            l.append("- veu passiva perifràstica (ser + participi)")
        else:
            l.append("- predicat verbal en veu activa")
        l += [f"- {NOMS[f]}" for f in e["funcions"]]
        if e["febles"]:
            l.append("- expressa el CD o el CI amb un pronom feble")
        return "\n".join(l)


    PLANTILLA = """Escriu {per} oracions simples en català sobre el tema «{tema}».
Han de ser oracions naturals, variades i correctes, amb una sola forma verbal (pot ser composta o perifràstica), sense subordinades ni coordinació d'oracions, i sense la paraula «no».
Cada oració ha de tenir EXACTAMENT aquests constituents, i cap altre complement:
{desc}
Respon només amb una llista JSON de {per} cadenes."""

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    peticions = []
    for _ in range(args.peticions):
        e = estructura_aleatoria()
        tema = rng.choice(TEMES)
        msg = [{"role": "user", "content": PLANTILLA.format(per=args.per, tema=tema, desc=descriu(e))}]
        peticions.append((e, tema, tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)))

    kw = dict(model=args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=4096, enable_prefix_caching=True, max_num_seqs=256)
    try:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as ex:
        print("Reintent sense limit_mm_per_prompt:", ex)
        llm = LLM(**kw)
    t0 = time.time()
    outs = llm.generate([p for _, _, p in peticions], SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1200))
    vistes, n = set(), 0
    with open(args.out, "w", encoding="utf-8") as f:
        for (e, tema, _), o in zip(peticions, outs):
            t = o.outputs[0].text
            t = t.rsplit("</think>", 1)[-1]
            m = re.search(r"\[.*\]", t, re.S)
            try:
                frases = json.loads(m.group(0)) if m else []
            except Exception:
                frases = []
            for fr in frases:
                if not isinstance(fr, str):
                    continue
                fr = fr.strip()
                if 3 <= len(fr.split()) <= 25 and fr.lower() not in vistes:
                    vistes.add(fr.lower()); n += 1
                    f.write(json.dumps({"frase": fr, "estructura_demanada": e, "tema": tema}, ensure_ascii=False) + "\n")
    print({"frases": n, "peticions": len(peticions), "segons": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
