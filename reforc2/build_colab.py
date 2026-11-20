# Genera reforc2_colab.ipynb: 3a ronda (2n reforç) — entrena 20k + reforç 1 + reforç 2 i avalua
import nbformat as nbf
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))
md("""
# Segon reforç · mini 20k + reforç 1 + reforç 2

Entrena l'alumne (Qwen3.5-0.8B) amb la mateixa recepta que la ronda anterior (TRL, 3 èpoques, lr 2e-5, lots de 16)
afegint-hi el reforç 2: 459 oracions escrites directament (sense el professor) sobre les categories on l'eval250 encara fallava.

**Entorn:** GPU A100 o L4 (Entorn d'execució → Canvia el tipus). Tarda uns 20-30 minuts.
""")
code("""
#@title Drive i dades
import os, json, random, subprocess, sys, shutil, glob
from google.colab import drive
drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/destillacio_sintaxi'
R2 = f'{BASE}/reforc2'; os.makedirs(R2, exist_ok=True)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
if not os.path.exists(f'{R2}/train_reforc2.jsonl'):
    from huggingface_hub import hf_hub_download
    shutil.copy(hf_hub_download('xaviviro/sintaxi-catala-destillada-20k-reforc2', 'reforc2.jsonl', repo_type='dataset'),
                f'{R2}/train_reforc2.jsonl')
print('reforç 2:', sum(1 for _ in open(f'{R2}/train_reforc2.jsonl')), 'oracions')
""")
code("""
#@title Instal·la TRL i companyia
!pip install -q -U trl datasets hf_transfer
!wget -q -O /content/common.py https://raw.githubusercontent.com/xaviviro/destillacio-sintaxi-catala/main/common.py
!wget -q -O /content/eval250.jsonl https://raw.githubusercontent.com/xaviviro/destillacio-sintaxi-catala/main/resultats/eval250.jsonl
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
sys.path.insert(0, '/content')
import importlib, common; importlib.reload(common)
from common import render, avalua, extreu_json, REFERENCIA
def llegeix(p): return [json.loads(l) for l in open(p, encoding='utf-8')]
""")
code("""
#@title Conjunt d'entrenament: 20k filtrat + reforç 1 (del Hub) + reforç 2
from huggingface_hub import hf_hub_download
TR1 = llegeix(hf_hub_download('xaviviro/sintaxi-catala-destillada-20k-reforc', 'train.jsonl', repo_type='dataset'))
VAL = llegeix(hf_hub_download('xaviviro/sintaxi-catala-destillada-20k-reforc', 'test.jsonl', repo_type='dataset'))
R2D = llegeix(f'{R2}/train_reforc2.jsonl')
REPETICIONS_R2 = 1   #@param {type:"integer"}
TRAIN = [{'frase': r['frase'], 'analisi': r['analisi']} for r in TR1] + \\
        [{'frase': r['frase'], 'analisi': r['analisi']} for r in R2D] * REPETICIONS_R2
random.Random(1).shuffle(TRAIN)
print(f'20k + reforç 1: {len(TR1)} · reforç 2: {len(R2D)} × {REPETICIONS_R2} · total {len(TRAIN)}')
""")
code("""
#@title Entrena (mateixa recepta que la ronda anterior)
ALUMNE = 'Qwen/Qwen3.5-0.8B'
MODEL_R2 = f'{R2}/alumne_destillat_20k_reforc2'
if not os.path.exists(f'{MODEL_R2}/config.json'):
    from transformers import AutoTokenizer
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    import torch, pandas as pd, gc
    tok = AutoTokenizer.from_pretrained(ALUMNE)
    FI = '<|im_end|>'
    def a_pc(rows):
        return Dataset.from_list([{'prompt': render(tok, r['frase'], False, curt=True),
                                   'completion': json.dumps(r['analisi'], ensure_ascii=False) + FI} for r in rows])
    cfg = SFTConfig(output_dir='/content/sft_r2', num_train_epochs=3, learning_rate=2e-5, lr_scheduler_type='cosine',
                    warmup_steps=20, per_device_train_batch_size=16, gradient_accumulation_steps=1,
                    bf16=True, logging_steps=10, eval_strategy='epoch', save_strategy='no', report_to='none',
                    max_length=1536, completion_only_loss=True, gradient_checkpointing=True,
                    model_init_kwargs={'torch_dtype': torch.bfloat16})
    trainer = SFTTrainer(model=ALUMNE, args=cfg, train_dataset=a_pc(TRAIN), eval_dataset=a_pc(VAL[:100]), processing_class=tok)
    trainer.train()
    trainer.model.to(torch.bfloat16); trainer.save_model(MODEL_R2); tok.save_pretrained(MODEL_R2)
    from huggingface_hub import snapshot_download
    orig = snapshot_download(ALUMNE, allow_patterns=['*.json', '*.jinja', '*.txt'])
    for f in os.listdir(orig):
        if f.endswith(('.json', '.jinja', '.txt')) and 'index' not in f and not os.path.exists(f'{MODEL_R2}/{f}'):
            shutil.copy(f'{orig}/{f}', MODEL_R2)
    pd.DataFrame(trainer.state.log_history).to_csv(f'{R2}/log_entrenament_20k_reforc2.csv', index=False)
    del trainer; gc.collect(); torch.cuda.empty_cache()
print(sorted(os.listdir(MODEL_R2)))
""")
code("""
#@title Avalua: eval250, referència de 30 i test (decodificació voraç)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL_R2); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(MODEL_R2, torch_dtype=torch.bfloat16, device_map='cuda').eval()
def analitza(frases, lot=64):
    out = []
    for i in range(0, len(frases), lot):
        p = [render(tok, f, False, curt=True) for f in frases[i:i + lot]]
        x = tok(p, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=768, do_sample=False)
        out += [extreu_json(tok.decode(s[x['input_ids'].shape[1]:], skip_special_tokens=True)) for s in y]
    return out
EV = llegeix('/content/eval250.jsonl')
pred = analitza([r['frase'] for r in EV])
with open(f'{R2}/pred_20k_reforc2_eval250.jsonl', 'w') as f:
    for r, a in zip(EV, pred): f.write(json.dumps({**r, 'analisi': a}, ensure_ascii=False) + '\\n')
print('eval250', avalua(pred, [r['ref'] for r in EV], [r['frase'] for r in EV]))
print({g: avalua([a for r, a in zip(EV, pred) if r['grup'] == g], [r['ref'] for r in EV if r['grup'] == g],
                 [r['frase'] for r in EV if r['grup'] == g])['exacte'] for g in dict.fromkeys(r['grup'] for r in EV)})
pr = analitza([f for f, _ in REFERENCIA]); print('ref30', avalua(pr, [a for _, a in REFERENCIA], [f for f, _ in REFERENCIA]))
pt = analitza([r['frase'] for r in VAL]); print('test300', avalua(pt, [r['analisi'] for r in VAL], [r['frase'] for r in VAL]))
""")
md("""
## Desa-ho

El model queda al Drive (`destillacio_sintaxi/reforc2/alumne_destillat_20k_reforc2`).
Opcional: puja'l al Hub amb el teu token (secret de Colab `HF_TOKEN`). El conjunt de dades ja és al Hub:
[xaviviro/sintaxi-catala-destillada-20k-reforc2](https://huggingface.co/datasets/xaviviro/sintaxi-catala-destillada-20k-reforc2).
""")
code("""
#@title (Opcional) Puja el model al Hub
PUJA = False   #@param {type:"boolean"}
if PUJA:
    from google.colab import userdata
    from huggingface_hub import HfApi
    api = HfApi(token=userdata.get('HF_TOKEN'))
    api.create_repo('xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc2', exist_ok=True)
    api.upload_folder(folder_path=MODEL_R2, repo_id='xaviviro/Qwen3.5-0.8B-sintaxi-catala-20k-reforc2')
    print('pujat')
""")
nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
               "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, "reforc2_colab.ipynb"); print("escrit reforc2_colab.ipynb", len(C), "cel·les")
