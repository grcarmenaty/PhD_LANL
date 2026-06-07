"""Generate notebooks/hires_all_gpu.ipynb — ONE notebook that runs the ENTIRE
hi-res study: every feature family (non-image -> vision) × every model × every
resolution × every task, with checkpoint/resume, skip-if-exists, pickup of past
results from all branches, and JSON-only autosave to colab-hires-all.
"""
from __future__ import annotations
import json
from pathlib import Path
from ml_pipeline.gen_zoo_notebooks import BOOTSTRAP, REGEN, PUSH, md, code

NB = Path(__file__).resolve().parent.parent / "notebooks" / "hires_all_gpu.ipynb"

SETUP = """import torch, numpy as np, h5py
from pathlib import Path
from ml_pipeline import hires_all as A, hires_tab as T, hires_zoo as Z
from ml_pipeline.tasks import build_targets
from ml_pipeline.train import make_split
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== CONFIG (edit me) =====================
TASKS = ['binary','col_location','mass_location','severity','type',
         'is_bolt','is_crack','is_mass','is_hole','is_pristine']
RESOLUTIONS = [128]                     # single reduced resolution (T4-friendly)
IMG_MODELS  = list(A.IMG_MODELS)        # cnn2d_shallow/deep, cnn3d, transformer, convnext_tiny, resnet50
TAB_MODELS  = list(A.TAB_MODELS)        # mlp, rf, xgb, cnn1d, transformer1d
CFDAC_FEATURES = list(Z.CFDAC_FEATURES) # 7 CFDAC channel-features
SUBSAMPLE = 3000
IMG_BATCH = 16        # T4 15 GB at 128 bins; raise on bigger GPUs / lower if OOM
TAB_BATCH = 256
VISION_SIZE = 224     # feed conv backbones at 224 (no upscaling a 128 grid)
PICKUP_BRANCHES = ['colab-hires-all']   # resume this run only (128 cells); add others to reuse them
FAMILY='all'; GH_RESULTS_BRANCH='colab-hires-all'; AUTOSAVE_GITHUB=True
CELLS = A.all_cells(TASKS, RESOLUTIONS, IMG_MODELS, TAB_MODELS, CFDAC_FEATURES)
print(len(CELLS),'cells queued across', len(RESOLUTIONS),'resolutions')
# Subset by editing the lists above, or e.g.:
#   CELLS = [('is_bolt','transformer1d','frf_realimag',1601)]
# ===========================================================

try:
    from google.colab import drive; drive.mount('/content/drive')
    OUT = Path('/content/drive/MyDrive/hires_cfdac/all')
except Exception:
    OUT = Path('results_hires_zoo_all')
OUT.mkdir(parents=True, exist_ok=True); (OUT/'cache').mkdir(exist_ok=True)

# Pick up already-trained cells (from PICKUP_BRANCHES; default = this run only).
import subprocess as _sp, os as _os
for _BR in PICKUP_BRANCHES:
    try:
        _sp.run(['git','-C','/content/PhD_LANL','fetch','--depth','1','origin',_BR], capture_output=True)
        _ls=_sp.run(['git','-C','/content/PhD_LANL','ls-tree','-r','--name-only','origin/'+_BR],capture_output=True,text=True).stdout
        (OUT/'per_case').mkdir(parents=True, exist_ok=True); _n=0
        for _l in _ls.splitlines():
            if '/per_case/' in _l and _l.endswith('.json'):
                _name=_os.path.basename(_l)
                if not (OUT/'per_case'/_name).exists():
                    _b=_sp.run(['git','-C','/content/PhD_LANL','show','origin/'+_BR+':'+_l],capture_output=True,text=True).stdout
                    if _b: (OUT/'per_case'/_name).write_text(_b); _n+=1
        if _n: print('picked up',_n,'cells from',_BR)
    except Exception as _e: print('pickup skip',_BR,_e)

# Image context (CFDAC computed on the fly; the engine decimates per resolution).
with h5py.File('dataset/features_hires.h5','r') as f:
    syn_tasks=build_targets(f['type_code'][:].astype('int64'),f['storey'][:].astype('int64'),
                            f['end'][:].astype('int64'),f['severity'][:].astype('float32'))
    H_ref_syn=torch.from_numpy(f['reference/frf_complex'][:].astype('complex64')).to(DEV)
with h5py.File('dataset/experimental_features_hires.h5','r') as f:
    exp_tasks=build_targets(f['type_code'][:].astype('int64'),f['storey'][:].astype('int64'),
                            f['end'][:].astype('int64'),f['severity'][:].astype('float32'))
    exp_names=[str(s) for s in f['names'][:]]
    H_ref_exp=torch.from_numpy(f['reference/frf_complex'][:].astype('complex64')).to(DEV)
with h5py.File('dataset/experimental_features_hires.h5','r') as f:
    H_exp=(f['frf_real'][:]+1j*f['frf_imag'][:]).astype('complex64')

# Tab feature caches per (feature, resolution) actually needed (built once each).
CACHE={}
for (ft,res) in sorted(A.features_needed(CELLS)):
    Xs=T.build_feature_cache('dataset/features_hires.h5',ft,OUT/'cache'/f'{ft}_syn_r{res}.npy',res=res)
    Xe=T.build_feature_cache('dataset/experimental_features_hires.h5',ft,OUT/'cache'/f'{ft}_exp_r{res}.npy',res=res)
    CACHE[(ft,res)]=(Xs,Xe); print('cache',ft,'@',res,'->',tuple(Xs.shape))

ctx=dict(out=OUT,dev=DEV,syn_tasks=syn_tasks,exp_tasks=exp_tasks,H_ref_syn=H_ref_syn,H_ref_exp=H_ref_exp,
         H_exp=H_exp,exp_names=exp_names,make_split=make_split,subsample=SUBSAMPLE,img_batch=IMG_BATCH,
         tab_batch=TAB_BATCH,vision_size=VISION_SIZE,syn_h5='dataset/features_hires.h5',
         exp_h5='dataset/experimental_features_hires.h5',tab_cache=CACHE)
print('context ready | device',DEV,'| amp',Z._amp_dtype(DEV))"""

RUN = """import torch, os, shutil, subprocess, time as _t
def _tok():
    try:
        from google.colab import userdata; return userdata.get('GH_TOKEN')
    except Exception: return os.environ.get('GH_TOKEN')
GH_TOKEN=_tok()
if AUTOSAVE_GITHUB and not GH_TOKEN: print('AUTOSAVE on but no GH_TOKEN -> Drive only')

def git_autosave(msg):
    if not (AUTOSAVE_GITHUB and GH_TOKEN): return
    repo='/content/PhD_LANL'; dst=os.path.join(repo,'results_hires_zoo',FAMILY)
    os.makedirs(os.path.join(dst,'per_case'), exist_ok=True)
    if not getattr(git_autosave,'_merged',False):   # one-time: pull remote cells so a force-push never overwrites a fuller branch
        subprocess.run(['git','-C',repo,'fetch','--depth','1','origin',GH_RESULTS_BRANCH],capture_output=True)
        _rl=subprocess.run(['git','-C',repo,'ls-tree','-r','--name-only','origin/'+GH_RESULTS_BRANCH],capture_output=True,text=True).stdout
        for _l in _rl.splitlines():
            if '/per_case/' in _l and _l.endswith('.json'):
                _fp=os.path.join(str(OUT),'per_case',os.path.basename(_l))
                if not os.path.exists(_fp):
                    _bb=subprocess.run(['git','-C',repo,'show','origin/'+GH_RESULTS_BRANCH+':'+_l],capture_output=True,text=True).stdout
                    if _bb: open(_fp,'w').write(_bb)
        git_autosave._merged=True
    for fn in (os.listdir(os.path.join(OUT,'per_case')) if os.path.isdir(os.path.join(OUT,'per_case')) else []):
        if fn.endswith('.json'): shutil.copy(os.path.join(OUT,'per_case',fn), os.path.join(dst,'per_case',fn))
    for sj in ['synth_test_zoo.json','synth_test_tab.json']:
        if os.path.exists(os.path.join(OUT,sj)): shutil.copy(os.path.join(OUT,sj), os.path.join(dst,sj))
    cwd=os.getcwd(); os.chdir(repo)
    subprocess.run(['git','config','user.email','colab@gpu.run']); subprocess.run(['git','config','user.name','colab-gpu'])
    subprocess.run(['git','add','-f','results_hires_zoo/'+FAMILY])
    if subprocess.run(['git','diff','--cached','--quiet']).returncode!=0:
        subprocess.run(['git','commit','-q','-m',msg])
        url=f'https://{GH_TOKEN}@github.com/grcarmenaty/phd_lanl.git'; ok=False
        for _a in range(5):
            r=subprocess.run(['git','push','--force',url,'HEAD:'+GH_RESULTS_BRANCH],capture_output=True,text=True)
            if r.returncode==0: ok=True; break
            _t.sleep(4*(2**_a))
        print('  autosave:', 'pushed -> '+GH_RESULTS_BRANCH if ok else 'push failed (Drive has it): '+r.stderr[-120:])
    os.chdir(cwd)

for i,(task,model,feature,res) in enumerate(CELLS):
    try:
        A.run_one(task, model, feature, res, ctx=ctx)
        if i % 5 == 0: git_autosave(f'colab autosave [all]: {i+1}/{len(CELLS)} cells')
    except Exception as e:
        print('CELL FAILED', task, model, feature, res, '::', repr(e)[:160])
        if torch.cuda.is_available(): torch.cuda.empty_cache()
git_autosave('colab autosave [all]: final')
print('\\nqueue done')"""

SUMMARY = """import json, numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score, f1_score
best=defaultdict(lambda: (-1,None))
n=0
for p in sorted((OUT/'per_case').glob('*_hires*.json')):
    d=json.loads(p.read_text()); m=d['meta']; r=d['rows']; n+=1
    if m['kind']!='cls': continue
    yt=np.array([x['y_true'] for x in r]); yp=np.array([x['y_pred'] for x in r])
    bal=balanced_accuracy_score(yt,yp); k=(m['task'],m['n_target'])
    if bal>best[k][0]: best[k]=(bal,f\"{m['model']}/{m['feature']}\")
print('total per_case files:', n)
print('\\nBest exp balanced-acc per (task, resolution):')
tasks=sorted({k[0] for k in best}); res=sorted({k[1] for k in best})
print('task'.ljust(14)+''.join(str(r).rjust(10) for r in res))
for t in tasks:
    print(t.ljust(14)+''.join((f'{best[(t,r)][0]:.3f}' if (t,r) in best else '  -- ').rjust(10) for r in res))
import shutil; shutil.make_archive('/content/results_all','zip',str(OUT))
try:
    from google.colab import files; files.download('/content/results_all.zip')
except Exception as e: print('zip at /content/results_all.zip', e)"""


def main():
    cells = [
        md("# CFDAC @128 — THE WHOLE STUDY in one notebook (T4-friendly)\n\n"
           "Every feature family (modal / indicators / FRF / timeseries → CFDAC images) and "
           "every model (MLP / RF / XGB / 1-D CNN+transformer → 2-D CNN shallow/deep / 3-D CNN "
           "/ CFDAC transformer / ConvNeXt-T / ResNet50) at **128-bin resolution**, across all "
           "10 tasks = **570 cells**. One unified grid dispatched to the right engine per cell. "
           "Sized for a **T4 (15 GB)**: 128² CFDAC is small, `IMG_BATCH=16`, vision fed at 224. "
           "Trains to convergence with checkpoint/resume, skips finished cells, autosaves JSON "
           "to `colab-hires-all` (5× retry). Set a GPU runtime + `GH_TOKEN` secret. Edit the "
           "CONFIG lists / `RESOLUTIONS` (e.g. add 400, 1601) to widen on a bigger GPU."),
        md("## 1 · Bootstrap"), code(BOOTSTRAP),
        md("## 2 · Regenerate the 1601-bin features"), code(REGEN),
        md("## 3 · Config + context + caches (edit the CONFIG block)"), code(SETUP),
        md("## 4 · Run the whole grid (resume + skip-if-exists + pickup)"), code(RUN),
        md("## 5 · Summary: best exp balanced-acc per (task × resolution) + zip"), code(SUMMARY),
        md("## 6 · (Optional) force-push the JSON snapshot now"), code(PUSH),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    NB.write_text(json.dumps(nb, indent=1))
    print("wrote", NB)


if __name__ == "__main__":
    main()
