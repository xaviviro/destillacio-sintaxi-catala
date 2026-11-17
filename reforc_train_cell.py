#@title 2a ronda · entrena el mini 20k + reforç i l'avalua
TRAIN20 = llegeix(f'{BASE}/train.jsonl'); REFORC = llegeix(f'{BASE}/reforc/train_reforc.jsonl')
TEST = llegeix(f'{BASE}/test.jsonl')
print('train 20k:', len(TRAIN20), '· reforç:', len(REFORC))
TRAIN_R = TRAIN20 + REFORC; random.Random(1).shuffle(TRAIN_R)
MODEL_R = f'{BASE}/alumne_destillat_20k_reforc'
if not os.path.exists(f'{MODEL_R}/config.json'):
    from transformers import AutoTokenizer
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    import torch, shutil, gc
    tok = AutoTokenizer.from_pretrained(ALUMNE)
    FI = '<|im_end|>'
    def a_pc(rows):
        return Dataset.from_list([{'prompt': render(tok, r['frase'], False, curt=True),
                                   'completion': json.dumps(r['analisi'], ensure_ascii=False) + FI} for r in rows])
    cfg = SFTConfig(output_dir='/content/sft_r', num_train_epochs=3, learning_rate=2e-5, lr_scheduler_type='cosine',
                    warmup_steps=20, per_device_train_batch_size=16, gradient_accumulation_steps=1,
                    bf16=True, logging_steps=10, eval_strategy='epoch', save_strategy='no', report_to='none',
                    max_length=1536, completion_only_loss=True, gradient_checkpointing=True,
                    model_init_kwargs={'torch_dtype': torch.bfloat16})
    trainer = SFTTrainer(model=ALUMNE, args=cfg, train_dataset=a_pc(TRAIN_R), eval_dataset=a_pc(TEST[:100]), processing_class=tok)
    trainer.train()
    trainer.save_model(MODEL_R); tok.save_pretrained(MODEL_R)
    from huggingface_hub import snapshot_download
    orig = snapshot_download(ALUMNE, allow_patterns=['*.json', '*.jinja', '*.txt'])
    for f in os.listdir(orig):
        if f.endswith(('.json', '.jinja', '.txt')) and 'safetensors' not in f and not os.path.exists(f'{MODEL_R}/{f}'):
            shutil.copy(f'{orig}/{f}', MODEL_R)
    pd.DataFrame(trainer.state.log_history).to_csv(f'{BASE}/log_entrenament_20k_reforc.csv', index=False)
    del trainer; gc.collect(); torch.cuda.empty_cache()
escriu('/content/test_inp.jsonl', [{'frase': r['frase'], 'ref': r['analisi']} for r in TEST])
for conj, inp in [('eval250', '/content/eval250.jsonl'), ('ref', REF_PATH), ('test', '/content/test_inp.jsonl')]:
    out = f'{EV_DIR}/pred_20k_reforc.jsonl' if conj == 'eval250' else f'{BASE}/{conj}_alumne_20k_reforc.jsonl'
    if not os.path.exists(out):
        corre(['/content/infer.py', '--model', MODEL_R, '--inp', inp, '--out', out, '--gpu_mem', '0.5', '--curt', '1'])
    rs = llegeix(out)
    print(conj, avalua([x['analisi'] for x in rs], [x['ref'] for x in rs], [x['frase'] for x in rs]))
rs = llegeix(f'{EV_DIR}/pred_20k_reforc.jsonl')
print({g: avalua([x['analisi'] for x in rs if x['grup']==g], [x['ref'] for x in rs if x['grup']==g], [x['frase'] for x in rs if x['grup']==g])['exacte'] for g in dict.fromkeys(x['grup'] for x in rs)})
print('FET2')