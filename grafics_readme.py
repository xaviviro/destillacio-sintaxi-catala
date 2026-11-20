# grafics_readme.py — genera les gràfiques del README a partir de resultats/resultats.json
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = json.load(open("resultats/resultats.json", encoding="utf-8"))
M = R["models"]
mides = [k for k in ("4k", "10k", "20k") if k in M]
COL = {"base": "#E45756", "4k": "#F58518", "10k": "#54A24B", "20k": "#2E8B57", "professor": "#4C78A8",
       "4k_q4": "#F8B26A", "10k_q4": "#8FD18A", "20k_q4": "#6FB98F"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def barres(conj, fitxer, titol, models):
    mets = ["exacte", "F1_complements", "subjecte", "nucli_verbal"]
    etiq = ["anàlisi exacta", "F1 complements", "subjecte", "nucli verbal"]
    models = [m for m in models if conj in M[m]]
    fig, ax = plt.subplots(figsize=(10, 4.6)); n = len(models); w = 0.8 / n; x = np.arange(len(mets))
    for i, m in enumerate(models):
        b = ax.bar(x + (i - (n - 1) / 2) * w, [M[m][conj][k] for k in mets], w, label=M[m]["nom"], color=COL[m])
        ax.bar_label(b, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, etiq); ax.set_ylim(0, 112); ax.set_ylabel("%")
    ax.set_title(titol, loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=n, fontsize=9)
    plt.tight_layout(); plt.savefig(fitxer, dpi=160); plt.close()


barres("ref", "resultats/efectivitat_referencia.png", "Referència manual · 30 oracions anotades a mà", ["base"] + mides + ["professor"])
barres("test", "resultats/efectivitat_test.png", "Test · 300 oracions (etiquetes del professor)", ["base"] + mides)

# corba de dades
fig, ax = plt.subplots(figsize=(8, 4.4))
xs = [0] + [M[m]["exemples_train"] for m in mides]
for conj, est, nom in (("ref", "-o", "anàlisi exacta · referència manual"), ("test", "--s", "anàlisi exacta · test 300")):
    ys = [M["base"][conj]["exacte"]] + [M[m][conj]["exacte"] for m in mides]
    ax.plot(xs, ys, est, color="#54A24B" if conj == "ref" else "#999", lw=2, label=nom)
    for x, y in zip(xs, ys): ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
# reforços: mateixes oracions del 20k + exemples dirigits als errors
for k, nom, col in (("20k_reforc", "+ reforç 1", "#72B7B2"), ("20k_reforc2", "+ reforç 2", "#2E5E4E")):
    if k in R:
        n = R[k]["n_train"]
        ax.plot([n], [R[k]["ref_exacte"]], "o", color=col, ms=9, zorder=4)
        ax.plot([n], [R[k]["test_exacte"]], "s", color=col, ms=7, zorder=4, mfc="white")
        ax.annotate(f"{nom}: {R[k]['ref_exacte']:.0f}", (n, R[k]["ref_exacte"]), textcoords="offset points",
                    xytext=(-8, 8), ha="right", fontsize=9, color=col, fontweight="bold")
ax.axhline(M["professor"]["ref"]["exacte"], color="#4C78A8", ls=":", lw=1.5)
ax.text(xs[0] + 50, M["professor"]["ref"]["exacte"] + 1.5, "professor 27B (referència manual)", color="#4C78A8", ha="left", fontsize=9)
ax.set_xticks(xs, ["base\n(0)"] + [f"{m}\n({M[m]['exemples_train']:,} ex.)".replace(",", ".") for m in mides])
ax.set_ylim(-5, 105); ax.set_ylabel("% anàlisi exacta")
ax.set_title("Més dades ajuden cada cop menys; dades dirigides, molt més", loc="left", fontsize=13, fontweight="bold")
ax.legend(frameon=False, loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig("resultats/corba_dades.png", dpi=160); plt.close()

# mida vs encert
fig, ax = plt.subplots(figsize=(8.5, 4.8))
punts = [("base", "base bf16")] + [(m, f"{m} bf16") for m in mides] + [(f"{m}_q4", f"{m} Q4") for m in mides if f"{m}_q4" in M] + [("professor", "professor 27B bf16")]
if "20k_reforc2" in R:
    M["20k_reforc2"] = {"mida_GB": M["20k"]["mida_GB"], "ref": {"exacte": R["20k_reforc2"]["ref_exacte"]}}
    M["20k_reforc2_mlx"] = {"mida_GB": 0.44, "ref": {"exacte": R["20k_reforc2"].get("ref_exacte_mlx4", R["20k_reforc2"]["ref_exacte"])}}
    COL.update({"20k_reforc2": "#2E5E4E", "20k_reforc2_mlx": "#8FC1A9"})
    punts += [("20k_reforc2", "+ 2 reforços bf16"), ("20k_reforc2_mlx", "+ 2 reforços MLX 4 bits")]
for k, et in punts:
    x, y = M[k]["mida_GB"], M[k]["ref"]["exacte"]
    ax.scatter(x, y, s=140, color=COL[k], edgecolor="black", lw=0.5, zorder=3)
    ax.annotate(et, (x, y), textcoords="offset points", xytext=(8, -16) if "reforc2" in k else (8, -4 if "q4" in k else 6), fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("mida dels fitxers del model (GB, escala logarítmica)"); ax.set_ylabel("% anàlisi exacta (referència manual)")
ax.set_xlim(0.3, 120); ax.set_ylim(-8, 105)
ax.set_title("Mida contra encert: fins a 100 vegades més petit", loc="left", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("resultats/mida_vs_encert.png", dpi=160); plt.close()

# mides i memòria
fig, ax = plt.subplots(figsize=(8.5, 3.8))
fil = [("professor 27B · fitxers bf16", M["professor"]["mida_GB"], COL["professor"]),
       ("mini 0,8B · fitxers bf16", M[mides[-1]]["mida_GB"], COL[mides[-1]]),
       ("mini 0,8B · GPU bf16 (pic, transformers)", R["memoria"]["mini_bf16_pic_GB_transformers"], COL[mides[-1]]),
       ("mini 0,8B · fitxer Q4_K_M", M["4k_q4"]["mida_GB"], COL["4k_q4"]),
       ("mini 0,8B · GPU Q4 (llama.cpp, ctx 2048)", R["memoria"]["mini_q4_gpu_GB_llamacpp_ctx2048"], COL["4k_q4"])]
y = np.arange(len(fil))[::-1]
b = ax.barh(y, [f[1] for f in fil], color=[f[2] for f in fil])
for yi, f in zip(y, fil): ax.text(f[1] * 1.15, yi, f"{f[1]:.2f} GB", va="center", fontsize=9)
ax.set_yticks(y, [f[0] for f in fil]); ax.set_xscale("log"); ax.set_xlim(0.2, 300); ax.set_xlabel("GB (escala logarítmica)")
ax.set_title("Mida i memòria", loc="left", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig("resultats/mides_memoria.png", dpi=160); plt.close()

# cost per oració
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
ms = ["base", mides[-1], "professor"]
axs[0].bar([M[m]["nom"] for m in ms], [M[m]["tokens_prompt"] for m in ms], color=[COL[m] for m in ms])
for i, m in enumerate(ms): axs[0].text(i, M[m]["tokens_prompt"] + 10, f"{M[m]['tokens_prompt']:.0f}", ha="center", fontsize=9)
axs[0].set_title("Tokens d'entrada per oració", loc="left", fontweight="bold")
axs[1].bar([M[m]["nom"] for m in ms], [M[m]["ms_frase_vllm_lot"] for m in ms], color=[COL[m] for m in ms]); axs[1].set_yscale("log")
for i, m in enumerate(ms): axs[1].text(i, M[m]["ms_frase_vllm_lot"] * 1.15, f"{M[m]['ms_frase_vllm_lot']:.0f} ms", ha="center", fontsize=9)
axs[1].set_title("ms per oració (lot de 30, vLLM)", loc="left", fontweight="bold")
for a in axs: a.tick_params(axis="x", labelsize=8)
plt.tight_layout(); plt.savefig("resultats/cost_per_oracio.png", dpi=160); plt.close()
print("gràfiques fetes")
