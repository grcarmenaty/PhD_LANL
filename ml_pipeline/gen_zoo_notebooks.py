"""Generate the Colab GPU notebooks for the hi-res CFDAC zoo.

Emits three thin notebooks under notebooks/ that all import the shared
engine ml_pipeline/hires_zoo.py and differ only in the model family:

  hires_cnn_zoo_gpu.ipynb     -> bespoke CNNs (cnn2d_deep, cnn2d_shallow, cnn3d)
  hires_transformer_gpu.ipynb -> CFDAC conv-tokenised Transformer
  hires_vision_gpu.ipynb      -> timm vision backbones (resnet50, convnext, effnet, swin, vit)

Each runs (model x 7 CFDAC features x 10 tasks), skip-if-exists, with results
persisted to Google Drive (survives Colab sessions). Set CELLS to a single
(task, model, feature) tuple to run exactly one cell per session.

Re-run this script to regenerate the notebooks after editing the template.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

PIP = ("subprocess.run([sys.executable,'-m','pip','-q','install',"
       "'timm','h5py','scikit-learn','pint','pyFRF','audiomentations'])")

BOOTSTRAP = f"""import os, sys, subprocess
GH_USER='grcarmenaty'; WORK='/content'; os.chdir(WORK)
def _tok():
    try:
        from google.colab import userdata; return userdata.get('GH_TOKEN')
    except Exception: return os.environ.get('GH_TOKEN')
def clone(repo, branch, dst):
    if os.path.isdir(dst): print('exists', dst); return
    t=_tok(); auth=f'{{t}}@' if t else ''
    url=f'https://{{auth}}github.com/{{GH_USER}}/{{repo}}.git'
    assert subprocess.run(['git','clone','--depth','1','-b',branch,url,dst]).returncode==0, \\
        f'clone failed {{repo}}@{{branch}} (private? add a GH_TOKEN Colab secret)'
clone('phd_lanl','main','/content/PhD_LANL')
clone('pymodal','master','/content/pymodal')   # sibling dir the scripts expect
for p in ('/content/PhD_LANL','/content/pymodal'):
    if p not in sys.path: sys.path.insert(0,p)
os.chdir('/content/PhD_LANL')
{PIP}
import torch
print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU - set a GPU runtime!')"""

REGEN = """import subprocess, sys, os, glob, json, h5py, numpy as np
from pathlib import Path
REPO=Path(os.getcwd())
def run(cmd): print('>>',' '.join(cmd)); assert subprocess.run(cmd).returncode==0, cmd
if not (REPO/'dataset'/'features_hires.h5').exists():
    run([sys.executable,'ml_pipeline/generate_dataset.py','--out','dataset_hires','--n-t','4096','--fs','256'])
    run([sys.executable,'ml_pipeline/build_hires_synth_features.py'])
if not (REPO/'experimental_frfs.h5').exists():
    with open('experimental_frfs.h5','wb') as o:
        for p in sorted(glob.glob('experimental_frfs_chunks/experimental_frfs.h5.part_*')):
            o.write(open(p,'rb').read())
if not (REPO/'dataset'/'experimental_features.h5').exists():
    from ml_pipeline.evaluate import primary_op
    with h5py.File('experimental_frfs.h5','r') as f: names=json.loads(f.attrs['case_names_json'])
    n=len(names); tc=np.zeros(n,np.int8); st=np.full(n,-1,np.int8); en=np.full(n,-1,np.int8); sv=np.zeros(n,np.float32)
    for i,nm in enumerate(names):
        op=primary_op(nm); tc[i]=op['type_code']; st[i]=op['storey']; en[i]=op['end']; sv[i]=op['severity']
    dt=h5py.string_dtype('utf-8')
    with h5py.File('dataset/experimental_features.h5','w') as o:
        o.create_dataset('names',data=np.array(names,dtype=object),dtype=dt)
        o.create_dataset('type_code',data=tc); o.create_dataset('storey',data=st)
        o.create_dataset('end',data=en); o.create_dataset('severity',data=sv)
if not (REPO/'dataset'/'experimental_features_hires.h5').exists():
    run([sys.executable,'ml_pipeline/build_hires_exp_features.py'])
print('features ready')"""

def SETUP(models_expr, family_dir):
    return f"""import torch, numpy as np, h5py
from pathlib import Path
from ml_pipeline import hires_zoo as Z
from ml_pipeline.tasks import build_targets
from ml_pipeline.train import make_split
DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== CONFIG (edit me) =====================
MODELS   = {models_expr}
TASKS    = ['binary','col_location','mass_location','severity','type',
            'is_bolt','is_crack','is_mass','is_hole','is_pristine']
FEATURES = list(Z.CFDAC_FEATURES)          # all 7 CFDAC channel-features
MAX_EPOCHS = 80        # safety cap; training stops early at convergence
PATIENCE   = 8         # early-stop after this many epochs with no val gain
SUBSAMPLE= 3000
BATCH    = 16          # lower to 8 if CUDA OOM
VISION_SIZE = 384      # conv vision backbones feed at this size (swin/vit fixed 224)
# --- GitHub autosave (each finished cell -> a per-family results branch) ---
FAMILY            = '{family_dir}'
AUTOSAVE_GITHUB   = True                               # set False to disable
GH_RESULTS_BRANCH = 'colab-hires-{family_dir}'         # never touches main
# Full grid below. To run ONE cell this session set e.g.:
#   CELLS = [('type','transformer','cfdac_realimag')]
CELLS = Z.all_cfdac_cells(MODELS, TASKS, FEATURES)
print(len(CELLS),'cells queued across', MODELS)
# ============================================================

# Persistence: Google Drive survives Colab session resets (skip-if-exists resumes).
try:
    from google.colab import drive; drive.mount('/content/drive')
    OUT=Path('/content/drive/MyDrive/hires_cfdac/{family_dir}')
except Exception:
    OUT=Path('results_hires_zoo_{family_dir}')
OUT.mkdir(parents=True, exist_ok=True); print('OUT =', OUT)

SYN=Path('dataset/features_hires.h5'); EXP=Path('dataset/experimental_features_hires.h5')
with h5py.File(SYN,'r') as f:
    syn_tasks=build_targets(f['type_code'][:].astype('int64'),f['storey'][:].astype('int64'),
                            f['end'][:].astype('int64'),f['severity'][:].astype('float32'))
    H_ref_syn=torch.from_numpy(f['reference/frf_complex'][:].astype('complex64')).to(DEV)
with h5py.File(EXP,'r') as f:
    exp_tasks=build_targets(f['type_code'][:].astype('int64'),f['storey'][:].astype('int64'),
                            f['end'][:].astype('int64'),f['severity'][:].astype('float32'))
    exp_names=[str(s) for s in f['names'][:]]
    H_ref_exp=torch.from_numpy(f['reference/frf_complex'][:].astype('complex64')).to(DEV)
with h5py.File(EXP,'r') as f:
    H_exp=(f['frf_real'][:]+1j*f['frf_imag'][:]).astype('complex64')
print('context ready; exp FRFs', H_exp.shape)"""

RUN = """import torch, os, shutil, subprocess
def _tok():
    try:
        from google.colab import userdata; return userdata.get('GH_TOKEN')
    except Exception: return os.environ.get('GH_TOKEN')
GH_TOKEN = _tok()
if AUTOSAVE_GITHUB and not GH_TOKEN:
    print('AUTOSAVE_GITHUB is on but no GH_TOKEN secret found -> results go to Drive/zip only.')

def git_autosave(msg):
    \"\"\"Force-push the JSON results of THIS family to its own results branch.
    Only per_case/*.json + synth_test_zoo.json are pushed (NOT the model
    .ckpt/.pt weights, which stay on Drive). Per-family branch => never
    conflicts with main or other families; always the full accumulated state.\"\"\"
    if not (AUTOSAVE_GITHUB and GH_TOKEN): return
    repo='/content/PhD_LANL'; dst=os.path.join(repo,'results_hires_zoo',FAMILY)
    os.makedirs(os.path.join(dst,'per_case'), exist_ok=True)
    # copy ONLY the json artefacts (skip the large models/ dir)
    for fn in os.listdir(os.path.join(OUT,'per_case')) if os.path.isdir(os.path.join(OUT,'per_case')) else []:
        if fn.endswith('.json'): shutil.copy(os.path.join(OUT,'per_case',fn), os.path.join(dst,'per_case',fn))
    if os.path.exists(os.path.join(OUT,'synth_test_zoo.json')):
        shutil.copy(os.path.join(OUT,'synth_test_zoo.json'), os.path.join(dst,'synth_test_zoo.json'))
    cwd=os.getcwd(); os.chdir(repo)
    subprocess.run(['git','config','user.email','colab@gpu.run'])
    subprocess.run(['git','config','user.name','colab-gpu'])
    subprocess.run(['git','add','-f',f'results_hires_zoo/{FAMILY}/per_case',f'results_hires_zoo/{FAMILY}/synth_test_zoo.json'])
    if subprocess.run(['git','diff','--cached','--quiet']).returncode!=0:
        subprocess.run(['git','commit','-q','-m',msg])
        url=f'https://{GH_TOKEN}@github.com/grcarmenaty/phd_lanl.git'
        r=subprocess.run(['git','push','--force',url,f'HEAD:{GH_RESULTS_BRANCH}'],
                         capture_output=True,text=True)
        print('  autosave:', f'pushed -> {GH_RESULTS_BRANCH}' if r.returncode==0
              else 'FAILED '+r.stderr[-160:])
    os.chdir(cwd)

for (task, model, feature) in CELLS:
    try:
        Z.run_cell(task, model, feature, syn_h5=SYN, exp_h5=EXP, out_dir=OUT, dev=DEV,
                   syn_tasks=syn_tasks, exp_tasks=exp_tasks, H_ref_syn=H_ref_syn,
                   H_ref_exp=H_ref_exp, H_exp=H_exp, exp_names=exp_names,
                   make_split=make_split, subsample=SUBSAMPLE, batch=BATCH,
                   vision_size=VISION_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE)
        git_autosave(f'colab autosave [{FAMILY}]: {task}/{model}/{feature}')
    except Exception as e:
        print('CELL FAILED', task, model, feature, '::', repr(e)[:200])
        if torch.cuda.is_available(): torch.cuda.empty_cache()
print('\\nqueue done')"""

SUMMARY = """import json, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
print(f\"{'cell':<48}{'kind':>5}{'synth':>8}{'expMF1/R2':>11}{'expBal':>8}{'expAcc':>8}{'collapse':>9}\")
print('-'*97)
for p in sorted((OUT/'per_case').glob('*_hires1601.json')):
    d=json.loads(p.read_text()); m=d['meta']; r=d['rows']
    yt=np.array([x['y_true'] for x in r]); yp=np.array([x['y_pred'] for x in r])
    name=f\"{m['task']}/{m['model']}/{m['feature']}\"
    if m['kind']=='cls':
        n=m['n_out']; bal=balanced_accuracy_score(yt,yp)
        mf1=f1_score(yt,yp,labels=list(range(n)),average='macro',zero_division=0); acc=accuracy_score(yt,yp)
        coll=(len(set(yp.tolist()))<=1) or (bal<=1/n+0.02)
        print(f\"{name:<48}{'cls':>5}{(m.get('synth_test_macro_f1') or 0):>8.3f}{mf1:>11.3f}{bal:>8.3f}{acc:>8.3f}{str(coll):>9}\")
    else:
        yt=yt.astype(float); yp=yp.astype(float); ss=np.sum((yt-yp)**2); st=np.sum((yt-yt.mean())**2)
        r2=1-ss/st if st>0 else 0; mae=np.mean(np.abs(yt-yp))
        print(f\"{name:<48}{'reg':>5}{(m.get('synth_test_metric') or 0):>8.3f}{r2:>11.3f}{'-':>8}{'-':>8}{'MAE=%.3f'%mae:>9}\")

# Zip for download (Drive already persists across sessions).
import shutil
z=str(OUT).rstrip('/').split('/')[-1]
shutil.make_archive('/content/'+z,'zip',str(OUT))
try:
    from google.colab import files; files.download('/content/'+z+'.zip')
except Exception as e: print('zip at /content/'+z+'.zip', e)"""

PUSH = """# Optional: push the JSON results to the repo (needs a GH_TOKEN secret with write).
import os, subprocess, shutil
tok=None
try:
    from google.colab import userdata; tok=userdata.get('GH_TOKEN')
except Exception: tok=os.environ.get('GH_TOKEN')
if not tok:
    print('No GH_TOKEN - hand the downloaded zip to the agent instead.')
else:
    dst='/content/PhD_LANL/results_hires_zoo'; os.makedirs(dst, exist_ok=True)
    if str(OUT)!=dst and (OUT/'per_case').exists():
        shutil.copytree(OUT, dst, dirs_exist_ok=True)
    os.chdir('/content/PhD_LANL')
    subprocess.run(['git','config','user.email','colab@gpu.run'])
    subprocess.run(['git','config','user.name','colab-gpu'])
    subprocess.run(['git','add','results_hires_zoo'])
    subprocess.run(['git','commit','-m','hires zoo (GPU): CFDAC cells synth+exp'])
    url=f'https://{tok}@github.com/grcarmenaty/phd_lanl.git'
    print(subprocess.run(['git','push',url,'HEAD:main'],capture_output=True,text=True).stderr[-400:])"""


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


def make_nb(title, intro, models_expr, family_dir):
    cells = [
        md(f"# {title}\n\n{intro}\n\n"
           "**No GPU? Set Runtime → Change runtime type → GPU.** Private repos: add a Colab "
           "secret `GH_TOKEN`. Results persist to Google Drive; **File → Save a copy in GitHub** "
           "saves this notebook itself. Set `CELLS` to one tuple to run a single cell per session."),
        md("## 1 · Bootstrap (clone phd_lanl + pymodal, install deps incl. pint/pyFRF/audiomentations)"),
        code(BOOTSTRAP),
        md("## 2 · Regenerate the 1601-bin features (gitignored; rebuilt from committed sources)"),
        code(REGEN),
        md("## 3 · Config + context (edit the CONFIG block)"),
        code(SETUP(models_expr, family_dir)),
        md("## 4 · Run the cell grid (skip-if-exists; resumes from Drive)"),
        code(RUN),
        md("## 5 · Honest summary (balanced-acc / macro-F1 / collapse) + zip download"),
        code(SUMMARY),
        md("## 6 · (Optional) push JSON results back to the repo"),
        code(PUSH),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


NOTEBOOKS = {
    "hires_cnn_zoo_gpu.ipynb": (
        "Hi-res 1601² CFDAC — bespoke CNN zoo (GPU)",
        "Trains the bespoke CNN families on full-1601² CFDAC across all 7 CFDAC "
        "channel-features and all 10 tasks: `cnn2d_deep` (ResNet18-style, consumes the "
        "full grid), `cnn2d_shallow` (the 128-baseline architecture, for parity), and "
        "`cnn3d` (channels as a depth axis).",
        "['cnn2d_deep','cnn2d_shallow','cnn3d']", "cnn"),
    "hires_transformer_gpu.ipynb": (
        "Hi-res 1601² CFDAC — Transformer (GPU)",
        "Trains a conv-tokenised Transformer (`CFDACTransformer`: strided-conv tokeniser "
        "→ Transformer encoder → cls head) on full-1601² CFDAC across all 7 features and "
        "10 tasks — tokenises the full resolution instead of resizing to 224.",
        "['transformer']", "transformer"),
    "hires_vision_gpu.ipynb": (
        "Hi-res 1601² CFDAC — vision backbones (GPU)",
        "Trains ImageNet-pretrained timm backbones (ResNet50, ConvNeXt-T, EfficientNet-B0, "
        "Swin-T, ViT-B/16) on CFDAC across all 7 features and 10 tasks. Conv backbones feed "
        "at `VISION_SIZE` (default 384, >> the 224 baseline); Swin/ViT are fixed at 224. "
        "**Large grid (5×7×10) — pick one backbone or a few features per session.**",
        "list(Z.VISION_BACKBONES)", "vision"),
}


def main():
    NB_DIR.mkdir(exist_ok=True)
    for fname, (title, intro, models_expr, fdir) in NOTEBOOKS.items():
        nb = make_nb(title, intro, models_expr, fdir)
        (NB_DIR / fname).write_text(json.dumps(nb, indent=1))
        print("wrote", NB_DIR / fname)


if __name__ == "__main__":
    main()
