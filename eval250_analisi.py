# eval250_analisi.py — mètriques i gràfics de l'avaluació independent de 250 oracions
import json, os, sys
sys.path.insert(0, "/content"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import avalua
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NOMS = [("base", "mini base\n(sense destil·lar)"), ("4k", "mini 4k"), ("10k", "mini 10k"),
        ("20k", "mini 20k"),
        ("20k_reforc", "mini 20k\n+ reforç"), ("20k_reforc2", "mini 20k\n+ 2 reforços"), ("professor", "professor\nQwen3.8-27B")]
COLS = {"base": "#E45756", "4k": "#F58518", "10k": "#54A24B", "20k": "#4C78A8", "20k_reforc": "#72B7B2", "20k_reforc2": "#2E5E4E", "professor": "#B279A2"}
METS = [("exacte", "anàlisi\nexacta"), ("F1_complements", "F1\ncomplements"), ("subjecte", "subjecte"),
        ("nucli_verbal", "nucli\nverbal"), ("tipus_i_veu", "tipus\ni veu")]


def llegeix(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def calcula(ev_dir):
    res = {}
    for nom, _ in NOMS:
        p = f"{ev_dir}/pred_{nom}.jsonl"
        if not os.path.exists(p):
            continue
        rs = llegeix(p)
        tot = avalua([r["analisi"] for r in rs], [r["ref"] for r in rs], [r["frase"] for r in rs])
        grups = {}
        for g in dict.fromkeys(r["grup"] for r in rs):
            sub = [r for r in rs if r["grup"] == g]
            grups[g] = dict(avalua([r["analisi"] for r in sub], [r["ref"] for r in sub], [r["frase"] for r in sub]), n=len(sub))
        st = json.load(open(p + ".stats.json"))
        res[nom] = {"global": tot, "grups": grups, "ms_per_frase": st["ms_per_frase"],
                    "tokens_prompt_mitjana": st["tokens_prompt_mitjana"]}
    json.dump(res, open(f"{ev_dir}/resultats_eval250.json", "w"), ensure_ascii=False, indent=1)
    return res


def grafic_global(res, path):
    noms = [(n, e) for n, e in NOMS if n in res]
    x = np.arange(len(METS)); w = 0.8 / len(noms)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k, (n, e) in enumerate(noms):
        v = [res[n]["global"][m] for m, _ in METS]
        b = ax.bar(x + (k - (len(noms) - 1) / 2) * w, v, w, label=e.replace("\n", " "), color=COLS[n])
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, val + 1, f"{val:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [e for _, e in METS]); ax.set_ylim(0, 108); ax.set_ylabel("%")
    ax.set_title("Avaluació independent: 250 oracions escrites i anotades a mà (eval250)")
    ax.legend(ncol=len(noms), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def grafic_grups(res, path, metrica="exacte"):
    noms = [(n, e) for n, e in NOMS if n in res]
    grups = list(res[noms[0][0]]["grups"])
    M = np.array([[res[n]["grups"][g][metrica] for n, _ in noms] for g in grups])
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(noms)), [e for _, e in noms], fontsize=8.5)
    ax.set_yticks(range(len(grups)), [f"{g} ({res[noms[0][0]]['grups'][g]['n']})" for g in grups], fontsize=9)
    for i in range(len(grups)):
        for j in range(len(noms)):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="black" if 25 < M[i, j] < 85 else "white")
    ax.set_title("eval250 · anàlisi exacta (%) per tipus d'oració")
    fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def taula_md(res):
    noms = [(n, e.replace("\n", " ")) for n, e in NOMS if n in res]
    cap = "| model | anàlisi exacta | F1 complements | F1 funcions | subjecte | nucli verbal | tipus i veu | JSON vàlid | ms/oració |\n|---|---|---|---|---|---|---|---|---|\n"
    files = [f"| {e} | **{res[n]['global']['exacte']:.1f}** | {res[n]['global']['F1_complements']:.1f} | {res[n]['global']['F1_funcions']:.1f} | "
             f"{res[n]['global']['subjecte']:.1f} | {res[n]['global']['nucli_verbal']:.1f} | {res[n]['global']['tipus_i_veu']:.1f} | "
             f"{res[n]['global']['json_valid']:.1f} | {res[n]['ms_per_frase']} |" for n, e in noms]
    return cap + "\n".join(files)


def taula_grups_md(res):
    noms = [(n, e.replace("\n", " ")) for n, e in NOMS if n in res]
    grups = list(res[noms[0][0]]["grups"])
    cap = "| tipus d'oració (n) | " + " | ".join(e for _, e in noms) + " |\n|---|" + "---|" * len(noms) + "\n"
    return cap + "\n".join(f"| {g} ({res[noms[0][0]]['grups'][g]['n']}) | " +
                           " | ".join(f"{res[n]['grups'][g]['exacte']:.0f}" for n, _ in noms) + " |" for g in grups)


if __name__ == "__main__":
    d = sys.argv[1]
    r = calcula(d)
    grafic_global(r, f"{d}/eval250_global.png"); grafic_grups(r, f"{d}/eval250_grups.png")
    print(taula_md(r)); print(); print(taula_grups_md(r))
