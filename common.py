# common.py — codi compartit: prompt, validació, mètriques i referència manual
import json, re, unicodedata, html

FUNCIONS = ["CD", "CI", "ATRIBUT", "CPRED", "CRV", "CAG",
            "CC_TEMPS", "CC_LLOC", "CC_MANERA", "CC_QUANTITAT",
            "CC_COMPANYIA", "CC_INSTRUMENT", "CC_CAUSA", "CC_FINALITAT"]

SISTEMA = """Ets un analitzador sintàctic del català. Analitza l'oració simple que et donin segons la gramàtica escolar catalana i respon NOMÉS amb un objecte JSON (sense cap text abans ni després) amb aquest format:
{"subjecte": {"text": <text literal del subjecte, o null si no n'hi ha>, "nucli": <nucli del subjecte, o null>, "eliptic": <true si el subjecte està omès>},
 "predicat": {"tipus": "verbal" o "nominal", "nucli": <forma verbal completa, amb auxiliars i perífrasis>,
              "complements": [{"funcio": <FUNCIÓ>, "text": <text literal>}]},
 "veu": "activa" o "passiva",
 "impersonal": <true o false>}

FUNCIÓ ha de ser una d'aquestes: CD, CI, ATRIBUT, CPRED, CRV, CAG, CC_TEMPS, CC_LLOC, CC_MANERA, CC_QUANTITAT, CC_COMPANYIA, CC_INSTRUMENT, CC_CAUSA, CC_FINALITAT.

Regles:
- Tots els textos han de ser fragments literals de l'oració, sense la puntuació final.
- Els pronoms febles (em, et, el, la, li, els, ens, us, hi, en...) són complements i s'etiqueten amb la seva funció.
- En «haver-hi», el «hi» forma part del nucli verbal («hi ha»).
- Predicat nominal: verb copulatiu (ser, estar, semblar) + ATRIBUT.
- Oració impersonal: subjecte amb text null, nucli null i eliptic false.
- Subjecte el·líptic: text null, nucli null i eliptic true.

Exemple — Oració: Ahir la Marta va comprar pa al forn.
{"subjecte": {"text": "la Marta", "nucli": "Marta", "eliptic": false}, "predicat": {"tipus": "verbal", "nucli": "va comprar", "complements": [{"funcio": "CC_TEMPS", "text": "Ahir"}, {"funcio": "CD", "text": "pa"}, {"funcio": "CC_LLOC", "text": "al forn"}]}, "veu": "activa", "impersonal": false}"""


def render(tok, frase, thinking=False, curt=False):
    # curt=True: només l'oració, sense instruccions ni exemple (així respon el model destil·lat)
    msgs = [{"role": "user", "content": frase}] if curt else \
           [{"role": "system", "content": SISTEMA}, {"role": "user", "content": f"Oració: {frase}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def extreu_json(text):
    if text is None:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None


def _base(t):
    return unicodedata.normalize("NFC", t).replace("’", "'").replace("`", "'").lower()


def norm(t):
    if t is None:
        return None
    t = re.sub(r"\s+", " ", _base(str(t))).strip()
    return t.strip(" .,;:!?¡¿\"«»")


def _ocurrencies(frase_n, frag):
    pre = "" if frag.startswith("'") else r"(?<!\w)"
    post = "" if frag.endswith("'") else r"(?!\w)"
    return [(m.start(), m.end()) for m in re.finditer(pre + re.escape(frag) + post, frase_n)]


def assigna_spans(frase, fragments):
    """Troba una posició no solapada per a cada fragment. Retorna llista de spans o None."""
    fn = re.sub(r"\s", " ", _base(frase))
    occ = [_ocurrencies(fn, norm(f)) if norm(f) else [] for f in fragments]
    if any(len(o) == 0 for o in occ):
        return None
    ordre = sorted(range(len(fragments)), key=lambda i: len(occ[i]))
    res = [None] * len(fragments)

    def bt(k, usats):
        if k == len(ordre):
            return True
        i = ordre[k]
        for s, e in occ[i]:
            if all(e <= a or s >= b for a, b in usats):
                res[i] = (s, e)
                if bt(k + 1, usats + [(s, e)]):
                    return True
        return False
    return res if bt(0, []) else None


def valida(frase, a):
    """Comprovacions deterministes. Retorna (ok, motiu)."""
    try:
        if not isinstance(a, dict):
            return False, "no és un objecte"
        s, p = a["subjecte"], a["predicat"]
        if a["veu"] not in ("activa", "passiva"):
            return False, "veu"
        if not isinstance(a["impersonal"], bool) or not isinstance(s["eliptic"], bool):
            return False, "booleans"
        if p["tipus"] not in ("verbal", "nominal"):
            return False, "tipus predicat"
        if not p.get("nucli"):
            return False, "sense nucli verbal"
        comps = p["complements"]
        if not isinstance(comps, list):
            return False, "complements"
        frags = [p["nucli"]]
        for c in comps:
            if c.get("funcio") not in FUNCIONS:
                return False, f"funció desconeguda {c.get('funcio')}"
            if not c.get("text"):
                return False, "complement buit"
            frags.append(c["text"])
        if s["text"] is None:
            if s["nucli"] is not None:
                return False, "nucli sense subjecte"
            if a["impersonal"] == s["eliptic"]:
                return False, "sense subjecte: ha de ser impersonal XOR el·líptic"
        else:
            if a["impersonal"] or s["eliptic"]:
                return False, "subjecte explícit però marcat impersonal/el·líptic"
            if not s["nucli"] or norm(s["nucli"]) not in norm(s["text"]):
                return False, "nucli del subjecte fora del subjecte"
            frags.append(s["text"])
        fs = [c["funcio"] for c in comps]
        if (p["tipus"] == "nominal") != ("ATRIBUT" in fs):
            return False, "nominal ⇔ ATRIBUT"
        if "CAG" in fs and a["veu"] != "passiva":
            return False, "CAG sense passiva"
        if assigna_spans(frase, frags) is None:
            return False, "fragments no literals o solapats"
        return True, "ok"
    except Exception as e:
        return False, f"estructura ({type(e).__name__})"


def estructura(a):
    s = a["subjecte"]
    tipus_s = "impersonal" if a["impersonal"] else ("eliptic" if s["eliptic"] else "explicit")
    return {"subjecte": tipus_s, "tipus": a["predicat"]["tipus"], "veu": a["veu"],
            "funcions": sorted(set(c["funcio"] for c in a["predicat"]["complements"]))}


def _parells(a):
    return {(c["funcio"], norm(c["text"])) for c in a["predicat"]["complements"]}


def avalua(prediccions, referencies, frases):
    n = len(referencies)
    m = dict(json_valid=0, analisi_valida=0, subjecte=0, nucli_verbal=0, tipus_i_veu=0, exacte=0)
    tp = fp = fn = 0
    ltp = lfp = lfn = 0
    for pred, ref, fr in zip(prediccions, referencies, frases):
        rp = _parells(ref)
        rl = [f for f, _ in rp]
        if not isinstance(pred, dict):
            fn += len(rp); lfn += len(rl)
            continue
        m["json_valid"] += 1
        ok, _ = valida(fr, pred)
        m["analisi_valida"] += ok
        try:
            sp, sr = pred["subjecte"], ref["subjecte"]
            subj = norm(sp.get("text")) == norm(sr["text"]) and bool(sp.get("eliptic")) == sr["eliptic"]
            nuc = norm(pred["predicat"].get("nucli")) == norm(ref["predicat"]["nucli"])
            tv = pred["predicat"].get("tipus") == ref["predicat"]["tipus"] and pred.get("veu") == ref["veu"]
            pp = _parells(pred)
        except Exception:
            subj = nuc = tv = False
            pp = set()
        m["subjecte"] += subj; m["nucli_verbal"] += nuc; m["tipus_i_veu"] += tv
        m["exacte"] += subj and nuc and tv and pp == rp
        tp += len(pp & rp); fp += len(pp - rp); fn += len(rp - pp)
        pl = sorted(f for f, _ in pp)
        rl2 = sorted(rl)
        comu = sum(min(pl.count(x), rl2.count(x)) for x in set(pl) | set(rl2))
        ltp += comu; lfp += len(pl) - comu; lfn += len(rl2) - comu
    res = {k: round(100 * v / n, 1) for k, v in m.items()}
    f1 = lambda a, b, c: round(100 * 2 * a / max(1, 2 * a + b + c), 1)
    res["F1_complements"] = f1(tp, fp, fn)
    res["F1_funcions"] = f1(ltp, lfp, lfn)
    res["n"] = n
    return res


COLORS = {"SUBJ": "#4C78A8", "NV": "#E45756", "CD": "#F58518", "CI": "#54A24B", "ATRIBUT": "#B279A2",
          "CPRED": "#9D755D", "CRV": "#72B7B2", "CAG": "#FF9DA6"}


def mostra_html(frase, a):
    """Pinta l'oració amb els constituents marcats."""
    if not isinstance(a, dict):
        return f"<p>{html.escape(frase)} <b style='color:#c00'>→ sortida no analitzable</b></p>"
    try:
        etiquetes, frags = [], []
        if a["subjecte"]["text"]:
            etiquetes.append("SUBJ"); frags.append(a["subjecte"]["text"])
        etiquetes.append("NV"); frags.append(a["predicat"]["nucli"])
        for c in a["predicat"]["complements"]:
            etiquetes.append(c["funcio"]); frags.append(c["text"])
        spans = assigna_spans(frase, frags)
    except Exception:
        spans = None
    if spans is None:
        return (f"<p>{html.escape(frase)} <b style='color:#c00'>→ anàlisi incoherent</b><br>"
                f"<code>{html.escape(json.dumps(a, ensure_ascii=False))[:400]}</code></p>")
    out, pos = [], 0
    for (s, e), et in sorted(zip(spans, etiquetes)):
        out.append(html.escape(frase[pos:s]))
        col = COLORS.get(et, "#888")
        out.append(f"<span style='border-bottom:3px solid {col};padding:1px 2px'>{html.escape(frase[s:e])}"
                   f"<sub style='color:{col};font-weight:bold'>{et}</sub></span>")
        pos = e
    out.append(html.escape(frase[pos:]))
    extra = []
    if a["subjecte"]["eliptic"]: extra.append("subjecte el·líptic")
    if a["impersonal"]: extra.append("impersonal")
    extra.append(f"predicat {a['predicat']['tipus']}, veu {a['veu']}")
    return f"<p style='font-size:1.1em'>{''.join(out)} <i style='color:#666'>({'; '.join(extra)})</i></p>"


# ---------------------------------------------------------------------------
# Referència manual (30 oracions). Anotació feta a mà seguint la gramàtica
# escolar; és revisable — si hi discrepes, corregeix-la aquí.
# ---------------------------------------------------------------------------
def _A(subj, nucli_s, tipus, nv, comps, veu="activa", impersonal=False, eliptic=False):
    return {"subjecte": {"text": subj, "nucli": nucli_s, "eliptic": eliptic},
            "predicat": {"tipus": tipus, "nucli": nv,
                         "complements": [{"funcio": f, "text": t} for f, t in comps]},
            "veu": veu, "impersonal": impersonal}

REFERENCIA = [
    ("La Maria llegeix un llibre a la biblioteca.", _A("La Maria", "Maria", "verbal", "llegeix", [("CD", "un llibre"), ("CC_LLOC", "a la biblioteca")])),
    ("Els nens juguen al parc cada tarda.", _A("Els nens", "nens", "verbal", "juguen", [("CC_LLOC", "al parc"), ("CC_TEMPS", "cada tarda")])),
    ("El meu germà és metge.", _A("El meu germà", "germà", "nominal", "és", [("ATRIBUT", "metge")])),
    ("Demà anirem a la platja amb els cosins.", _A(None, None, "verbal", "anirem", [("CC_TEMPS", "Demà"), ("CC_LLOC", "a la platja"), ("CC_COMPANYIA", "amb els cosins")], eliptic=True)),
    ("L'àvia va regalar una bufanda a la Júlia.", _A("L'àvia", "àvia", "verbal", "va regalar", [("CD", "una bufanda"), ("CI", "a la Júlia")])),
    ("Li he enviat la carta aquest matí.", _A(None, None, "verbal", "he enviat", [("CI", "Li"), ("CD", "la carta"), ("CC_TEMPS", "aquest matí")], eliptic=True)),
    ("La casa va ser construïda pels avis.", _A("La casa", "casa", "verbal", "va ser construïda", [("CAG", "pels avis")], veu="passiva")),
    ("Plou molt a la muntanya.", _A(None, None, "verbal", "Plou", [("CC_QUANTITAT", "molt"), ("CC_LLOC", "a la muntanya")], impersonal=True)),
    ("Hi ha molta gent a la plaça.", _A(None, None, "verbal", "Hi ha", [("CD", "molta gent"), ("CC_LLOC", "a la plaça")], impersonal=True)),
    ("En Pere confia en els seus amics.", _A("En Pere", "Pere", "verbal", "confia", [("CRV", "en els seus amics")])),
    ("Els alumnes van arribar cansats a l'escola.", _A("Els alumnes", "alumnes", "verbal", "van arribar", [("CPRED", "cansats"), ("CC_LLOC", "a l'escola")])),
    ("Van nomenar la Laia delegada de classe.", _A(None, None, "verbal", "Van nomenar", [("CD", "la Laia"), ("CPRED", "delegada de classe")], eliptic=True)),
    ("La sopa està molt calenta.", _A("La sopa", "sopa", "nominal", "està", [("ATRIBUT", "molt calenta")])),
    ("El fuster talla la fusta amb una serra.", _A("El fuster", "fuster", "verbal", "talla", [("CD", "la fusta"), ("CC_INSTRUMENT", "amb una serra")])),
    ("Vam ajornar la festa per la pluja.", _A(None, None, "verbal", "Vam ajornar", [("CD", "la festa"), ("CC_CAUSA", "per la pluja")], eliptic=True)),
    ("Estudia cada dia per aprovar l'examen.", _A(None, None, "verbal", "Estudia", [("CC_TEMPS", "cada dia"), ("CC_FINALITAT", "per aprovar l'examen")], eliptic=True)),
    ("La Núria parla l'anglès perfectament.", _A("La Núria", "Núria", "verbal", "parla", [("CD", "l'anglès"), ("CC_MANERA", "perfectament")])),
    ("Els turistes mengen paella al port.", _A("Els turistes", "turistes", "verbal", "mengen", [("CD", "paella"), ("CC_LLOC", "al port")])),
    ("Aquest llibre m'agrada molt.", _A("Aquest llibre", "llibre", "verbal", "agrada", [("CI", "m'"), ("CC_QUANTITAT", "molt")])),
    ("La professora ens explica la lliçó amb paciència.", _A("La professora", "professora", "verbal", "explica", [("CI", "ens"), ("CD", "la lliçó"), ("CC_MANERA", "amb paciència")])),
    ("L'he vist al mercat.", _A(None, None, "verbal", "he vist", [("CD", "L'"), ("CC_LLOC", "al mercat")], eliptic=True)),
    ("Els veïns semblen contents.", _A("Els veïns", "veïns", "nominal", "semblen", [("ATRIBUT", "contents")])),
    ("El partit va ser guanyat per l'equip local.", _A("El partit", "partit", "verbal", "va ser guanyat", [("CAG", "per l'equip local")], veu="passiva")),
    ("Fa fred aquesta nit.", _A(None, None, "verbal", "Fa", [("CD", "fred"), ("CC_TEMPS", "aquesta nit")], impersonal=True)),
    ("La Marta pensa en les vacances.", _A("La Marta", "Marta", "verbal", "pensa", [("CRV", "en les vacances")])),
    ("El nen dorm tranquil al llit.", _A("El nen", "nen", "verbal", "dorm", [("CPRED", "tranquil"), ("CC_LLOC", "al llit")])),
    ("Ahir el meu pare va comprar pa i llet al forn.", _A("el meu pare", "pare", "verbal", "va comprar", [("CC_TEMPS", "Ahir"), ("CD", "pa i llet"), ("CC_LLOC", "al forn")])),
    ("Les flors del jardí són precioses.", _A("Les flors del jardí", "flors", "nominal", "són", [("ATRIBUT", "precioses")])),
    ("Escriurem una carta al director.", _A(None, None, "verbal", "Escriurem", [("CD", "una carta"), ("CI", "al director")], eliptic=True)),
    ("En Jordi treballa a Girona des de fa anys.", _A("En Jordi", "Jordi", "verbal", "treballa", [("CC_LLOC", "a Girona"), ("CC_TEMPS", "des de fa anys")])),
]
