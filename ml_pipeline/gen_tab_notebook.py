"""Generate notebooks/hires_tabular_gpu.ipynb — the non-CFDAC-image feature zoo
(modal / indicators / frf) with proper MLP / RF / XGB / 1-D CNN / transformer,
at 1601 resolution. Reuses the bootstrap/regen/push blocks from gen_zoo_notebooks.

Re-run to regenerate.
"""
from __future__ import annotations
import json
from pathlib import Path
from ml_pipeline.gen_zoo_notebooks import BOOTSTRAP, REGEN, PUSH, md, code

NB = Path(__file__).resolve().parent.parent / "notebooks" / "hires_tabular_gpu.ipynb"

SETUP = """import torch, numpy as np, h5py
from pathlib import Path
from ml_pipeline import hires_tab as T
from ml_pipeline.tasks import build_targets
from ml_pipeline.train import make_split
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== CONFIG (edit me) =====================
MODELS = ['mlp','rf','xgb','cnn1d','transformer1d']
TASKS  = ['binary','col_location','mass_location','severity','type',
          'is_bolt','is_crack','is_mass','is_hole','is_pristine']
SUBSAMPLE = 4000          # synth samples per task (raise toward 10000 for more data)
BATCH     = 256           # tabular/seq NN batch (tiny models -> large batch fine)
AUTOSAVE_GITHUB   = True
FAMILY            = 'tabular'
GH_RESULTS_BRANCH = 'colab-hires-tabular'
# feature/model compatibility lives in T.TAB_MODEL_FEATURES:
#   mlp: modal,indicators,frf_mag,frf_realimag | rf,xgb: modal,indicators
#   cnn1d,transformer1d: frf_mag,frf_realimag
# To run ONE cell: CELLS = [('is_hole','mlp','modal')]
CELLS = T.tab_cells(MODELS, TASKS)
print(len(CELLS),'cells queued across', MODELS)
# ===========================================================

try:
    from google.colab import drive; drive.mount('/content/drive')
    OUT = Path('/content/drive/MyDrive/hires_cfdac/tabular')
except Exception:
    OUT = Path('results_hires_zoo_tabular')
OUT.mkdir(parents=True, exist_ok=True); (OUT/'cache').mkdir(exist_ok=True)

SYN = 'dataset/features_hires.h5'; EXP = 'dataset/experimental_features_hires.h5'
with h5py.File(SYN,'r') as f:
    syn_tasks = build_targets(f['type_code'][:].astype('int64'),f['storey'][:].astype('int64'),
                              f['end'][:].astype('int64'),f['severity'][:].astype('float32'))
with h5py.File(EXP,'r') as f:
    exp_tasks = build_targets(f['type_code'][:].astype('int64'),f['storey'][:].astype('int64'),
                              f['end'][:].astype('int64'),f['severity'][:].astype('float32'))
    exp_names = [str(s) for s in f['names'][:]]

# Precompute each feature ONCE for ALL samples (cached on Drive), so indicators
# (a 1601 CFDAC per sample) are never recomputed per task.
feats_used = sorted({f for (_,_,f) in CELLS})
CACHE = {}
for ft in feats_used:
    print('building cache:', ft, '(indicators = slow: 1601 CFDAC/sample)')
    Xs = T.build_feature_cache(SYN, ft, OUT/'cache'/f'{ft}_syn.npy')
    Xe = T.build_feature_cache(EXP, ft, OUT/'cache'/f'{ft}_exp.npy')
    CACHE[ft] = (Xs, Xe)
print('caches:', {k:(tuple(v[0].shape),tuple(v[1].shape)) for k,v in CACHE.items()})
print('device', DEV, '| amp', T._amp_dtype(DEV))"""

RUN = """import torch, os, shutil, subprocess
def _tok():
    try:
        from google.colab import userdata; return userdata.get('GH_TOKEN')
    except Exception: return os.environ.get('GH_TOKEN')
GH_TOKEN = _tok()
if AUTOSAVE_GITHUB and not GH_TOKEN:
    print('AUTOSAVE on but no GH_TOKEN -> Drive/zip only')

def git_autosave(msg):
    if not (AUTOSAVE_GITHUB and GH_TOKEN): return
    repo='/content/PhD_LANL'; dst=os.path.join(repo,'results_hires_zoo',FAMILY)
    os.makedirs(os.path.join(dst,'per_case'), exist_ok=True)
    for fn in (os.listdir(os.path.join(OUT,'per_case')) if os.path.isdir(os.path.join(OUT,'per_case')) else []):
        if fn.endswith('.json'): shutil.copy(os.path.join(OUT,'per_case',fn), os.path.join(dst,'per_case',fn))
    if os.path.exists(os.path.join(OUT,'synth_test_tab.json')):
        shutil.copy(os.path.join(OUT,'synth_test_tab.json'), os.path.join(dst,'synth_test_tab.json'))
    cwd=os.getcwd(); os.chdir(repo)
    subprocess.run(['git','config','user.email','colab@gpu.run'])
    subprocess.run(['git','config','user.name','colab-gpu'])
    subprocess.run(['git','add','-f',f'results_hires_zoo/{FAMILY}/per_case',f'results_hires_zoo/{FAMILY}/synth_test_tab.json'])
    if subprocess.run(['git','diff','--cached','--quiet']).returncode!=0:
        subprocess.run(['git','commit','-q','-m',msg])
        url=f'https://{GH_TOKEN}@github.com/grcarmenaty/phd_lanl.git'
        r=subprocess.run(['git','push','--force',url,f'HEAD:{GH_RESULTS_BRANCH}'],capture_output=True,text=True)
        print('  autosave:', f'pushed -> {GH_RESULTS_BRANCH}' if r.returncode==0 else 'FAILED '+r.stderr[-160:])
    os.chdir(cwd)

for (task, model, feature) in CELLS:
    try:
        T.run_tab_cell(task, model, feature, out_dir=OUT, dev=DEV, syn_tasks=syn_tasks,
                       exp_tasks=exp_tasks, Xsyn=CACHE[feature][0], Xexp=CACHE[feature][1],
                       exp_names=exp_names, make_split=make_split, subsample=SUBSAMPLE, batch=BATCH)
        git_autosave(f'colab autosave [tabular]: {task}/{model}/{feature}')
    except Exception as e:
        print('CELL FAILED', task, model, feature, '::', repr(e)[:200])
        if torch.cuda.is_available(): torch.cuda.empty_cache()
print('\\nqueue done')"""

SUMMARY = """import json, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
print(f\"{'cell':<46}{'kind':>5}{'synth':>8}{'expMF1/R2':>11}{'expBal':>8}{'collapse':>9}\")
print('-'*86)
for p in sorted((OUT/'per_case').glob('*_hires1601.json')):
    d=json.loads(p.read_text()); m=d['meta']; r=d['rows']
    yt=np.array([x['y_true'] for x in r]); yp=np.array([x['y_pred'] for x in r])
    name=f\"{m['task']}/{m['model']}/{m['feature']}\"
    if m['kind']=='cls':
        n=m['n_out']; bal=balanced_accuracy_score(yt,yp)
        mf1=f1_score(yt,yp,labels=list(range(n)),average='macro',zero_division=0)
        coll=(len(set(yp.tolist()))<=1) or (bal<=1/n+0.02)
        print(f\"{name:<46}{'cls':>5}{(m.get('synth_test_macro_f1') or 0):>8.3f}{mf1:>11.3f}{bal:>8.3f}{str(coll):>9}\")
    else:
        yt=yt.astype(float); yp=yp.astype(float); ss=np.sum((yt-yp)**2); st=np.sum((yt-yt.mean())**2)
        r2=1-ss/st if st>0 else 0
        print(f\"{name:<46}{'reg':>5}{(m.get('synth_test_metric') or 0):>8.3f}{r2:>11.3f}{'-':>8}{'-':>9}\")
import shutil
shutil.make_archive('/content/results_tabular','zip',str(OUT))
try:
    from google.colab import files; files.download('/content/results_tabular.zip')
except Exception as e: print('zip at /content/results_tabular.zip', e)"""


def main():
    cells = [
        md("# Hi-res 1601 — non-image feature zoo (modal / indicators / FRF) · GPU\n\n"
           "The scientifically-sound complement to the CFDAC-image notebooks: the feature "
           "families that transferred **best** in the 128 baseline (modal, indicators) plus raw "
           "FRF, with **proper MLP / RandomForest / XGBoost / 1-D CNN / transformer**. Features "
           "are standardised on the synth-train fold and applied zero-shot to experiment; NN "
           "models train to convergence with checkpoint/resume; trees fit once. Set a GPU "
           "runtime; add a `GH_TOKEN` secret for autosave to `colab-hires-tabular`."),
        md("## 1 · Bootstrap"), code(BOOTSTRAP),
        md("## 2 · Regenerate 1601-bin features"), code(REGEN),
        md("## 3 · Config + precompute feature caches (once, on Drive)"), code(SETUP),
        md("## 4 · Run the grid (skip-if-exists; NN resume from checkpoint)"), code(RUN),
        md("## 5 · Honest summary + zip download"), code(SUMMARY),
        md("## 6 · (Optional) push results to the repo"), code(PUSH),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    NB.write_text(json.dumps(nb, indent=1))
    print("wrote", NB)


if __name__ == "__main__":
    main()
