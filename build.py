import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s, titol=None):
    s = s.strip("\n")
    if titol: s = f"#@title {titol}\n" + s
    meta = {"cellView": "form"} if s.startswith("#@title") else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": s})
PAL = ["#1f3a5f", "#7a2e3a", "#2f5d50", "#5b3f7a", "#7a5a1f", "#1f5a7a", "#3f3f3f", "#6b2f5f", "#2f4f7a", "#555"]
def slide(n, titol, sub, cos=""):
    md(f"""# {n:02d} · {titol}
### *{sub}*

---

{cos}""")

def wf(path, name): code(f"open('/content/{name}', 'w', encoding='utf-8').write(r\'\'\'" + open(path, encoding="utf-8").read() + "\'\'\')", titol=f"Escriu {name}")

slide(0, "Destil·lació: d'un Qwen gran a un Qwen petit", "Un analitzador sintàctic del català que cap a la butxaca", """
**La tasca.** Entra una oració simple en català i en surt l'anàlisi sintàctica escolar en JSON: subjecte, nucli verbal, CD, CI, atribut, CPRED, CRV, CAG, CC…

1. **El petit**, tal com surt de fàbrica → no se'n surt
2. **El professor** escriu oracions amb una estructura demanada i les analitza *a cegues*
3. **Filtre**: validació automàtica + coincidència amb l'estructura demanada
4. **Destil·lació**: fine-tuning del petit amb les dades bones
5. **Avaluació**: petit vs destil·lat vs professor, sobre 30 oracions anotades a mà

Tot es desa a Google Drive (`MyDrive/destillacio_sintaxi`): si el dataset o el model ja hi són, el notebook els reutilitza i se salta els passos lents.
""")

slide(1, "Preparació", "GPU, Google Drive i opcions")
code("""
#@title Opcions
REGENERAR_FRASES   = False   # True → el professor torna a escriure les oracions
REGENERAR_ANOTACIO = False   # True → el professor torna a analitzar-les
REENTRENAR         = False   # True → torna a entrenar l'alumne
PETICIONS          = 500     # × 8 oracions per petició
PENSAR_PROFESSOR   = True    # el professor raona (thinking) abans de respondre: més lent, més fiable

import os, json, glob, random, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/destillacio_sintaxi'
os.makedirs(BASE, exist_ok=True)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
print('Fitxers ja desats:', sorted(os.listdir(BASE)))
""")

md("### Instal·lació")
code("""
#@title Instal·la vLLM, TRL i companyia
!pip install -q -U vllm trl peft datasets hf_transfer
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
""")

slide(2, "Les regles del joc", "El mateix prompt per a tothom, validació automàtica i 30 oracions anotades a mà", "Les cel·les següents escriuen el codi compartit. Al **prompt** hi ha les instruccions i un sol exemple; la **validació** comprova que cada fragment sigui literal de l'oració, que no se solapin i que l'estructura sigui coherent (nominal ⇔ atribut, CAG ⇔ passiva…).")
wf("common.py", "common.py")
wf("infer.py", "infer.py")
wf("gen_frases.py", "gen_frases.py")
code("""
#@title Tria de models (el professor més gran que hi cap, l'alumne més petit)
import torch, importlib
sys.path.insert(0, '/content')
import common; importlib.reload(common)
from common import *
from huggingface_hub import HfApi
from IPython.display import HTML, display
import pandas as pd

api = HfApi()
def existeix(m):
    try: api.model_info(m); return True
    except Exception: return False

mem = torch.cuda.get_device_properties(0).total_memory / 1e9
cand_prof = (['Qwen/Qwen3.8-27B'] if mem > 75 else []) + \\
            ['Qwen/Qwen3.8-27B-FP8', 'Qwen/Qwen3.6-27B-FP8', 'Qwen/Qwen3.5-27B-FP8'] + \\
            (['Qwen/Qwen3.6-27B', 'Qwen/Qwen3.5-27B'] if mem > 75 else [])
cand_alum = ['Qwen/Qwen3.8-0.8B', 'Qwen/Qwen3.6-0.8B', 'Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3-0.6B']
PROFESSOR = next(m for m in cand_prof if existeix(m))
ALUMNE    = next(m for m in cand_alum if existeix(m))
print(f'GPU: {mem:.0f} GB\\nProfessor: {PROFESSOR}\\nAlumne:    {ALUMNE}')

def corre(args):
    # vLLM en un procés a part: quan acaba, la GPU queda lliure
    p = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print(p.stderr[-6000:]); raise RuntimeError('ha fallat')

def llegeix(path): return [json.loads(l) for l in open(path, encoding='utf-8')]
def escriu(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\\n')

REF_PATH = f'{BASE}/referencia_manual.jsonl'
escriu(REF_PATH, [{'frase': f, 'ref': a} for f, a in REFERENCIA])
""")

slide(3, "Abans de res: el petit, tal com ve", "Mateix prompt, mateixes 30 oracions. A veure què fa.")
code("""
#@title Executa el petit sobre les 30 oracions
corre(['/content/infer.py', '--model', ALUMNE, '--inp', REF_PATH, '--out', '/content/ref_alumne_base.jsonl', '--gpu_mem', '0.5'])
base_ref = llegeix('/content/ref_alumne_base.jsonl')
res_base = avalua([r['analisi'] for r in base_ref], [r['ref'] for r in base_ref], [r['frase'] for r in base_ref])
display(pd.DataFrame([res_base], index=[f'{ALUMNE} (base)']))
""")
code("""
#@title Uns quants exemples
for r in base_ref[:10]:
    display(HTML('<b>Referència:</b>' + mostra_html(r['frase'], r['ref']) + '<b>Petit base:</b>' + mostra_html(r['frase'], r['analisi'])))
""")

slide(4, "El professor fabrica el dataset", "Primer escriu oracions amb l'estructura que li demanem; després les analitza sense saber-la", "El truc és que la **generació** i l'**anàlisi** són independents: si l'anàlisi a cegues coincideix amb l'estructura que havíem demanat, tenim dues opinions que concorden. Si no, la descartem.")
code("""
#@title 4a · Oracions amb estructura demanada
FRASES = f'{BASE}/frases.jsonl'
if REGENERAR_FRASES or not os.path.exists(FRASES):
    corre(['/content/gen_frases.py', '--model', PROFESSOR, '--out', FRASES, '--peticions', str(PETICIONS)])
frases = llegeix(FRASES)
print(len(frases), 'oracions'); print(*random.sample([f['frase'] for f in frases], 8), sep='\\n')
""")
code("""
#@title 4b · El professor les analitza a cegues (sense saber l'estructura demanada)
ANOT = f'{BASE}/anotacio_professor.jsonl'
if REGENERAR_ANOTACIO or not os.path.exists(ANOT):
    corre(['/content/infer.py', '--model', PROFESSOR, '--inp', FRASES, '--out', ANOT,
           '--thinking', str(int(PENSAR_PROFESSOR)), '--max_tokens', '4096' if PENSAR_PROFESSOR else '768',
           '--temperature', '0.6' if PENSAR_PROFESSOR else '0', '--max_len', '8192'])
anot = llegeix(ANOT)
print(json.load(open(ANOT + '.stats.json')))
""")
code("""
#@title 4c · Filtre: validació determinista + coincidència amb l'estructura demanada
from collections import Counter
motius, bons = Counter(), []
for r in anot:
    a = r['analisi']
    if a is None: motius['JSON no vàlid'] += 1; continue
    ok, m = valida(r['frase'], a)
    if not ok: motius[m] += 1; continue
    dem = {k: v for k, v in r['estructura_demanada'].items() if k != 'febles'}
    if estructura(a) != dem: motius["no coincideix amb l'estructura demanada"] += 1; continue
    bons.append({'frase': r['frase'], 'analisi': a, 'estructura': dem})
print(f'{len(bons)}/{len(anot)} superen el filtre ({100*len(bons)/len(anot):.0f}%)')
display(pd.DataFrame(motius.most_common(), columns=['motiu de descart', 'n']))

# Treiem del dataset qualsevol oració de la referència manual
refs = {norm(f) for f, _ in REFERENCIA}
bons = [b for b in bons if norm(b['frase']) not in refs]
random.Random(0).shuffle(bons)
TEST, TRAIN = bons[:300], bons[300:]
escriu(f'{BASE}/train.jsonl', TRAIN); escriu(f'{BASE}/test.jsonl', TEST)
print('train', len(TRAIN), '· test', len(TEST))
display(pd.Series(Counter(f for b in TRAIN for f in b['estructura']['funcions'])).sort_values(ascending=False).to_frame('ocurrències al train'))
""")

slide(5, "El sostre", "Quant encerta el professor en les 30 oracions anotades a mà?", "L'alumne aprèn del professor: el que el professor no sap, l'alumne tampoc no ho aprendrà.")
code("""
#@title El professor analitza les 30 oracions
REF_PROF = f'{BASE}/ref_professor.jsonl'
if not os.path.exists(REF_PROF):
    corre(['/content/infer.py', '--model', PROFESSOR, '--inp', REF_PATH, '--out', REF_PROF,
           '--thinking', str(int(PENSAR_PROFESSOR)), '--max_tokens', '4096', '--temperature', '0.6' if PENSAR_PROFESSOR else '0'])
prof_ref = llegeix(REF_PROF)
res_prof = avalua([r['analisi'] for r in prof_ref], [r['ref'] for r in prof_ref], [r['frase'] for r in prof_ref])
display(pd.DataFrame([res_base, res_prof], index=[f'{ALUMNE} (base)', PROFESSOR]))
""")

slide(6, "Destil·lació", "Fine-tuning complet de l'alumne amb les anàlisis filtrades del professor", "Destil·lació per seqüència (SFT): l'alumne aprèn a donar la resposta final directament, **sense raonar**. El professor sí que ha raonat per arribar-hi.\n\nI una cosa més: l'alumne s'entrena **sense prompt**. Només li donem l'oració; les instruccions, el format i l'exemple queden *dins dels pesos*. El model destil·lat respon directament amb el JSON.")
code("""
#@title Entrena (o recupera de Drive) l'alumne destil·lat
MODEL_OUT = f'{BASE}/alumne_destillat'
if REENTRENAR or not os.path.exists(f'{MODEL_OUT}/config.json'):
    from transformers import AutoTokenizer
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    tok = AutoTokenizer.from_pretrained(ALUMNE)
    FI = '<|im_end|>'
    def a_pc(rows):
        return Dataset.from_list([{'prompt': render(tok, r['frase'], False, curt=True),
                                   'completion': json.dumps(r['analisi'], ensure_ascii=False) + FI} for r in rows])
    ds_train, ds_eval = a_pc(TRAIN), a_pc(TEST[:100])
    print(ds_train[0]['prompt'][-200:], ds_train[0]['completion'])
    cfg = SFTConfig(output_dir='/content/sft', num_train_epochs=3, learning_rate=2e-5, lr_scheduler_type='cosine',
                    warmup_ratio=0.05, per_device_train_batch_size=16, gradient_accumulation_steps=1,
                    bf16=True, logging_steps=10, eval_strategy='epoch', save_strategy='no', report_to='none',
                    max_length=1536, completion_only_loss=True, gradient_checkpointing=True,
                    model_init_kwargs={'torch_dtype': torch.bfloat16})
    trainer = SFTTrainer(model=ALUMNE, args=cfg, train_dataset=ds_train, eval_dataset=ds_eval, processing_class=tok)
    trainer.train()
    trainer.save_model(MODEL_OUT); tok.save_pretrained(MODEL_OUT)
    # vLLM necessita també els fitxers de configuració originals (processador, plantilla...)
    from huggingface_hub import snapshot_download
    import shutil
    orig = snapshot_download(ALUMNE, allow_patterns=['*.json', '*.jinja', '*.txt'])
    for f in os.listdir(orig):
        if not os.path.exists(f'{MODEL_OUT}/{f}'): shutil.copy(f'{orig}/{f}', MODEL_OUT)
    pd.DataFrame(trainer.state.log_history).to_csv(f'{BASE}/log_entrenament.csv', index=False)
    del trainer; import gc; gc.collect(); torch.cuda.empty_cache()
print(sorted(os.listdir(MODEL_OUT)))
""")

slide(7, "Resultats", "Petit base · petit destil·lat · professor")
code("""
#@title Avalua els tres models
TEST_PATH = f'{BASE}/test_inp.jsonl'
escriu(TEST_PATH, [{'frase': r['frase'], 'ref': r['analisi']} for r in TEST])
for nom, model in [('base', ALUMNE), ('destillat', MODEL_OUT)]:
    for conj, path in [('ref', REF_PATH), ('test', TEST_PATH)]:
        out = f'/content/{conj}_alumne_{nom}.jsonl'
        if not os.path.exists(out) or nom == 'destillat':
            corre(['/content/infer.py', '--model', model, '--inp', path, '--out', out, '--gpu_mem', '0.5'] + (['--curt', '1'] if nom == 'destillat' else []))

files = {}
def taula(conj):
    rows, idx = [], []
    for nom, path in [(f'{ALUMNE} base', f'/content/{conj}_alumne_base.jsonl'),
                      ('alumne destil·lat', f'/content/{conj}_alumne_destillat.jsonl')] + \\
                     ([(f'{PROFESSOR} (professor)', REF_PROF)] if conj == 'ref' else []):
        rs = llegeix(path); files[(conj, nom)] = rs
        r = avalua([x['analisi'] for x in rs], [x['ref'] for x in rs], [x['frase'] for x in rs])
        st = json.load(open(path + '.stats.json')); r['ms/frase'] = st['ms_per_frase']; r['tokens de prompt'] = st.get('tokens_prompt_mitjana')
        rows.append(r); idx.append(nom)
    return pd.DataFrame(rows, index=idx)

print('Referència manual (30 oracions, anotades a mà)'); display(taula('ref'))
print('Test (300 oracions, etiquetes del professor filtrades)'); display(taula('test'))
""")
code("""
import matplotlib.pyplot as plt, numpy as np
t = taula('ref')
mets = ['exacte', 'F1_complements', 'subjecte', 'nucli_verbal', 'analisi_valida']
noms = {'exacte': 'anàlisi\\nexacta', 'F1_complements': 'F1\\ncomplements', 'subjecte': 'subjecte', 'nucli_verbal': 'nucli\\nverbal', 'analisi_valida': 'sortida\\nvàlida'}
cols = ['#E45756', '#54A24B', '#4C78A8']
fig, ax = plt.subplots(figsize=(11, 4.8)); w = 0.26; x = np.arange(len(mets))
for i, (nom, fila) in enumerate(t.iterrows()):
    b = ax.bar(x + (i - 1) * w, [fila[m] for m in mets], w, label=f"{nom}  ·  {fila['ms/frase']:.0f} ms/frase", color=cols[i])
    ax.bar_label(b, fmt='%.0f', fontsize=9, padding=2)
ax.set_xticks(x, [noms[m] for m in mets]); ax.set_ylim(0, 110); ax.set_ylabel('%')
ax.set_title('Referència manual · 30 oracions', loc='left', fontsize=14, fontweight='bold')
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3)
plt.tight_layout(); plt.savefig(f'{BASE}/resultats.png', dpi=200); plt.show()
""", titol="Gràfic per a la diapositiva")
code("""
#@title Comparació visual a la referència manual
base_r = {r['frase']: r['analisi'] for r in llegeix('/content/ref_alumne_base.jsonl')}
dest_r = {r['frase']: r['analisi'] for r in llegeix('/content/ref_alumne_destillat.jsonl')}
for f, a in REFERENCIA:
    display(HTML(f"<hr><b>Referència</b>{mostra_html(f, a)}<b>Base</b>{mostra_html(f, base_r[f])}<b>Destil·lat</b>{mostra_html(f, dest_r[f])}"))
""")

slide(8, "Demo en directe", "Escriu una oració, sense cap instrucció, i el destil·lat respon amb el JSON")
code("""
#@title Carrega el model destil·lat
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
tok_d = AutoTokenizer.from_pretrained(MODEL_OUT)
try:
    mod_d = AutoModelForCausalLM.from_pretrained(MODEL_OUT, torch_dtype=torch.bfloat16).cuda().eval()
except Exception:
    from transformers import AutoModelForImageTextToText
    mod_d = AutoModelForImageTextToText.from_pretrained(MODEL_OUT, torch_dtype=torch.bfloat16).cuda().eval()

def analitza(frase):
    ids = tok_d(render(tok_d, frase, False, curt=True), return_tensors='pt').to('cuda')
    t0 = time.time()
    out = mod_d.generate(**ids, max_new_tokens=400, do_sample=False)
    txt = tok_d.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True)
    a = extreu_json(txt)
    display(HTML(mostra_html(frase, a) + f"<small>{1000*(time.time()-t0):.0f} ms</small>"))
    print(json.dumps(a, ensure_ascii=False, indent=1) if a else txt)
    return a
""")
code("""
#@title Analitza una oració
frase = 'La Laia va regalar un dibuix a la seva àvia per Nadal.'  #@param {type:"string"}
analitza(frase);
""")

slide(9, "Per endur-se'l", "(Opcional) Exportar a GGUF per fer-lo córrer al portàtil")
code("""
#@title Exporta a GGUF (descomenta per executar)
# !git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
# !pip install -q -r /content/llama.cpp/requirements.txt
# !python /content/llama.cpp/convert_hf_to_gguf.py {MODEL_OUT} --outtype q8_0 --outfile {BASE}/alumne_destillat_q8.gguf
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("destillacio_sintaxi_catala.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(cells), "cel·les")
