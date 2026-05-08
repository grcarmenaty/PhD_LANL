"""
setup_env.py  —  creates the Python virtual environment for 3SBB_exploration.ipynb

Run with:  py setup_env.py   (or python setup_env.py)
"""
import subprocess, sys, os
from pathlib import Path

HERE   = Path(__file__).parent.resolve()
VENV   = HERE / ".venv"
REQS   = HERE / "requirements.txt"
KERNEL = "3SBB"

SEP = "=" * 60

def run(cmd, **kw):
    print(f"  > {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result

print(SEP)
print(" 3SBB notebook  —  environment setup")
print(SEP)
print()

# ── 1. Show which Python is running this script ───────────────────────────────
print(f"[1/4] Using Python: {sys.executable}")
print(f"      Version     : {sys.version.split()[0]}")
print()

# ── 2. Create virtual environment ────────────────────────────────────────────
if (VENV / "Scripts" / "python.exe").exists():
    print(f"[2/4] .venv already exists  ({VENV})")
else:
    print(f"[2/4] Creating .venv at {VENV} ...")
    run([sys.executable, "-m", "venv", str(VENV)])
    print("      Done.")
print()

venv_python = VENV / "Scripts" / "python.exe"
venv_pip    = VENV / "Scripts" / "pip.exe"

# ── 3. Install packages ───────────────────────────────────────────────────────
print(f"[3/4] Installing packages from requirements.txt ...")
run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "-q"])
run([str(venv_python), "-m", "pip", "install", "-r", str(REQS)])
print("      Done.")
print()

# ── 4. Register Jupyter kernel ────────────────────────────────────────────────
print(f"[4/4] Registering Jupyter kernel '{KERNEL}' ...")
run([
    str(venv_python), "-m", "ipykernel", "install",
    "--user",
    "--name",         KERNEL,
    "--display-name", "Python (3SBB)",
])
print()

print(SEP)
print(" Setup complete!")
print(f" Virtual env : {VENV}")
print(f" Kernel name : Python (3SBB)")
print()
print(" Next step   : double-click run_notebook.bat")
print(SEP)
