# reforc.py — 2a ronda de destil·lació dirigida als punts febles de l'eval250 (CRV, impersonals, pronoms febles, CPRED)
# S'executa dins del notebook (fa servir BASE, PROFESSOR, ALUMNE, corre, llegeix, escriu, norm, valida, estructura, REFERENCIA)
CODI = r'''
RF = f'{BASE}/reforc'; os.makedirs(RF, exist_ok=True)
F_FR, F_AN = f'{RF}/frases.jsonl', f'{RF}/anotacio_professor.jsonl'
excl = {norm(r['frase']) for f in glob.glob(f'{BASE}/frases*.jsonl') for r in llegeix(f)}
excl |= {norm(r['frase']) for r in llegeix(f'{BASE}/eval250/eval250.jsonl')} | {norm(f) for f, _ in REFERENCIA}
excl |= {norm(r['frase']) for r in llegeix(f'{BASE}/test.jsonl')}
if not os.path.exists(F_FR):
    corre(['/content/gen_frases.py', '--model', PROFESSOR, '--out', '/content/tmp_reforc.jsonl',
           '--peticions', str(N_PETICIONS), '--seed', '9001', '--reforc', '1'])
    escriu(F_FR, [r for r in llegeix('/content/tmp_reforc.jsonl') if norm(r['frase']) not in excl])
print('oracions de reforç:', len(llegeix(F_FR)))
if not os.path.exists(F_AN):
    corre(['/content/infer.py', '--model', PROFESSOR, '--inp', F_FR, '--out', F_AN,
           '--thinking', '1', '--max_tokens', '4096', '--temperature', '0.6', '--max_len', '8192'])
from collections import Counter
motius, REFORC = Counter(), []
for r in llegeix(F_AN):
    a = r['analisi']
    if a is None: motius['JSON no vàlid'] += 1; continue
    ok, m = valida(r['frase'], a)
    if not ok: motius[m] += 1; continue
    dem = {k: v for k, v in r['estructura_demanada'].items() if k != 'febles'}
    if estructura(a) != dem: motius["no coincideix amb l'estructura demanada"] += 1; continue
    if norm(r['frase']) in excl: continue
    REFORC.append({'frase': r['frase'], 'analisi': a, 'estructura': dem})
escriu(f'{RF}/train_reforc.jsonl', REFORC)
print(f'{len(REFORC)} superen el filtre ·', motius.most_common(5))
print(Counter(f for b in REFORC for f in b['estructura']['funcions']).most_common(),
      Counter(b['estructura']['subjecte'] for b in REFORC))
'''
