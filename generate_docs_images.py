"""Generate figures for MODEL.md.

Builds:
  docs/images/building_pristine.png        — 3D view, no damage
  docs/images/building_d85_combined.png    — 3D view with bolt + mass
  docs/images/frf_pristine.png             — 9-sensor FRF comparison, Pristine
  docs/images/frf_d85_1ad_1bd.png          — 9-sensor FRF, D(85%) 1AD+1BD
  docs/images/cfdac_pristine.png           — exp / model CFDAC pair, Pristine
  docs/images/cfdac_d85_2bd.png            — exp / model CFDAC pair, D(85%) 2BD
  docs/images/sci_scoreboard.png           — SCI bar chart over all 61 cases
  docs/images/jsr_curve.png                — k_eff/k_ff vs J for symmetric and asymmetric joints
  docs/images/asymmetric_jsr.png           — heatmap k_eff/k_ff (J_t, J_b)
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from reduced_model_semirigid import _semirigid_factor

OUT_DIR = _HERE / 'docs' / 'images'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants for 3D rendering ─────────────────────────────────────────────
PL  = 0.305
PT  = 0.0254
CLY = 0.0064
CGAP = 0.0005
SH  = PT + 0.1524
z_pl = [k * SH + PT/2 for k in range(4)]
xl = 0.0254 / 2
xh = PL - xl
cx = PL / 2

SENSOR_POS = [
    (xl, PL, z_pl[0]), (xl, PL, z_pl[3]), (xl, PL, z_pl[2]),
    (xl, PL, z_pl[1]), (xh, PL, z_pl[0]), (xh, PL, z_pl[3]),
    (xh, PL, z_pl[2]), (xh, PL, z_pl[1]), (cx, 0., z_pl[0]),
]
SIG_NAMES = [2, 5, 6, 7, 8, 11, 12, 13, 14]
SHAKER_POS = (cx, 0., z_pl[0])
COL_XY = [
    (xl, -CLY/2 - CGAP), (xh, -CLY/2 - CGAP),
    (xl, PL + CLY/2 + CGAP), (xh, PL + CLY/2 + CGAP),
]


def _box_faces(x0, x1, y0, y1, z0, z1):
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return [[v[i] for i in idx]
            for idx in [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                         (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]]


def render_building(ax, *, bolt_storeys=(), mass_plates=()):
    """Render the 3SBB on *ax* (3D), highlighting damage where requested."""
    for k, zc in enumerate(z_pl):
        zb, zt = zc - PT/2, zc + PT/2
        ax.add_collection3d(Poly3DCollection(
            _box_faces(0, PL, 0, PL, zb, zt),
            alpha=0.20, facecolor='silver', edgecolor='gray', lw=0.4))
        ax.text(PL/2, PL/2, zt + 0.005,
                ['Base', 'Floor 1', 'Floor 2', 'Floor 3'][k],
                ha='center', va='bottom', fontsize=7, color='dimgray')

    for cxc, cyc in COL_XY:
        for st in range(3):
            colour = 'crimson' if st in bolt_storeys else 'steelblue'
            lw = 4.0 if st in bolt_storeys else 3.0
            ax.plot([cxc, cxc], [cyc, cyc],
                    [z_pl[st], z_pl[st+1]],
                    color=colour, lw=lw, solid_capstyle='round')

    for k in mass_plates:
        zc = z_pl[k] + PT/2
        m_h = 0.022
        ax.add_collection3d(Poly3DCollection(
            _box_faces(PL/2 - 0.04, PL/2 + 0.04,
                       PL/2 - 0.04, PL/2 + 0.04,
                       zc, zc + m_h),
            alpha=0.85, facecolor='goldenrod', edgecolor='black', lw=0.6))
        ax.text(PL/2, PL/2, zc + m_h + 0.005, 'Mass',
                ha='center', fontsize=6, color='darkgoldenrod')

    for (sx, sy, sz), nm in zip(SENSOR_POS, SIG_NAMES):
        ax.scatter([sx], [sy], [sz], s=42, c='tab:blue', marker='^',
                   zorder=6, depthshade=False)
        ax.text(sx, sy + 0.015, sz + 0.008, f'S{nm}', fontsize=6,
                color='tab:blue')

    sx, sy, sz = SHAKER_POS
    ax.scatter([sx], [sy], [sz], s=180, c='darkorange', marker='*',
               zorder=7, depthshade=False)
    ax.text(sx, sy - 0.04, sz + 0.005, 'shaker', ha='center',
            fontsize=6, color='darkorange')

    ax.set_xlim(0, PL); ax.set_ylim(-0.05, PL + 0.05)
    ax.set_zlim(0, z_pl[3] + 0.04)
    ax.set_xlabel('X (m)', labelpad=2, fontsize=7)
    ax.set_ylabel('Y (m)', labelpad=2, fontsize=7)
    ax.set_zlabel('Z (m)', labelpad=2, fontsize=7)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=20, azim=-55)


def fig_building_pristine():
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    render_building(ax)
    ax.set_title('LANL 3SBB — pristine geometry', fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'building_pristine.png', dpi=140,
                bbox_inches='tight')
    plt.close(fig)


def fig_building_combined():
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    render_building(ax, bolt_storeys=(0,), mass_plates=(0,))
    ax.set_title('Example: D(85%) 1BD + Mass Base damage indicators',
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'building_d85_combined.png', dpi=140,
                bbox_inches='tight')
    plt.close(fig)


# ── FRF + CFDAC plots ──────────────────────────────────────────────────────
GRID = [
    [1, 5, None],     # floor 3:  S5 (xl), S11 (xh)
    [2, 6, None],     # floor 2:  S6 (xl), S12 (xh)
    [3, 7, None],     # floor 1:  S7 (xl), S13 (xh)
    [0, 4,    8],     # base:     S2 (xl), S8 (xh), S14 (cx -Y)
]
ROW_LABEL = ['Floor 3', 'Floor 2', 'Floor 1', 'Base']


def _load_frfs():
    with h5py.File(_HERE / 'median_frfs.h5', 'r') as f:
        cn = [c.decode() for c in f['case_names'][:]]
        H_e = f['median_frf'][:]
        fr_e = f['freq'][:]
    with h5py.File(_HERE / 'synthetic_frfs.h5', 'r') as f:
        cns = [c.decode() for c in f['case_names'][:]]
        H_s = f['frfs'][:]
        fr_s = f['freqs'][:]
    return cn, H_e, fr_e, cns, H_s, fr_s


def fig_frf(case_name: str, fname: str):
    cn, H_e, fr_e, cns, H_s, fr_s = _load_frfs()
    H_exp = H_e[cn.index(case_name)]
    H_mod = H_s[cns.index(case_name)]

    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True)
    for r, row in enumerate(GRID):
        for c, ch in enumerate(row):
            ax = axes[r, c]
            if ch is None:
                ax.set_visible(False); continue
            ax.semilogy(fr_e, np.abs(H_exp[:, ch]),
                        color='tab:blue', lw=1.4, label='exp median')
            ax.semilogy(fr_s, np.abs(H_mod[:, ch]),
                        'r--', lw=1.1, label='model')
            ax.set_xlim(0, 100); ax.set_ylim(1e-3, 1e1)
            ax.grid(True, which='both', ls=':', alpha=0.3)
            ax.set_title(f'S{SIG_NAMES[ch]} — {ROW_LABEL[r]}', fontsize=10)
            if r == 3: ax.set_xlabel('Frequency (Hz)')
            if c == 0: ax.set_ylabel(r'$|H|$ (m/s²)/N')
            if r == 0 and c == 0:
                ax.legend(loc='lower right', fontsize=9)
    fig.suptitle(f'FRF — {case_name}', fontsize=12, y=0.995)
    plt.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=140, bbox_inches='tight')
    plt.close(fig)


def _band_indices(freq, lo=5.0, hi=100.0, step=1.0):
    target = np.arange(lo, hi + 1e-9, step)
    return np.array([int(np.argmin(np.abs(freq - t))) for t in target])


def cfdac(H):
    inner = H.conj() @ H.T
    d = np.real(np.diag(inner)).copy(); d[d < 1e-30] = 1e-30
    return (np.abs(inner) ** 2) / np.outer(d, d)


def sci(C1, C2):
    a = C1.ravel() - C1.mean(); b = C2.ravel() - C2.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) ** 2 / d ** 2) if d > 0 else 0.0


def fig_cfdac(case_name: str, fname: str):
    cn, H_e, fr_e, cns, H_s, fr_s = _load_frfs()
    ie = _band_indices(fr_e)
    isn = _band_indices(fr_s)
    band_freqs = fr_e[ie]

    Ce = cfdac(H_e[cn.index(case_name)][ie])
    Cs = cfdac(H_s[cns.index(case_name)][isn])
    s = sci(Ce, Cs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    extent = [band_freqs[0], band_freqs[-1], band_freqs[-1], band_freqs[0]]
    for ax, C, label in [(axes[0], Ce, 'experiment'),
                         (axes[1], Cs, 'model')]:
        im = ax.imshow(C, vmin=0, vmax=1, extent=extent,
                       cmap='magma', aspect='equal')
        ax.set_title(f'CFDAC — {label}', fontsize=11)
        ax.set_xlabel(r'$f_j$ (Hz)')
        ax.set_ylabel(r'$f_i$ (Hz)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f'{case_name}    SCI = {s:.3f}', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_sci_scoreboard():
    cn, H_e, fr_e, cns, H_s, fr_s = _load_frfs()
    ie = _band_indices(fr_e)
    isn = _band_indices(fr_s)
    sc = []
    for i, n in enumerate(cn):
        Ce = cfdac(H_e[i][ie])
        Cs = cfdac(H_s[cns.index(n)][isn])
        sc.append((sci(Ce, Cs), n))
    sc.sort()
    values = np.array([s for s, _ in sc])
    labels = [n for _, n in sc]
    mean_ = float(values.mean())

    fig, ax = plt.subplots(figsize=(11, 13))
    colors = ['crimson' if v < 0.85 else 'goldenrod' if v < 0.95 else 'steelblue'
              for v in values]
    ax.barh(np.arange(len(labels)), values, color=colors, alpha=0.85)
    ax.axvline(mean_, color='black', lw=1.2, ls='--',
               label=f'mean SCI = {mean_:.3f}')
    ax.axvline(0.95, color='steelblue', lw=0.8, ls=':', alpha=0.7)
    ax.axvline(0.90, color='goldenrod', lw=0.8, ls=':', alpha=0.7)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('SCI (1 = perfect CFDAC match)')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title(f'SCI scoreboard — {len(values)} IQS cases\n'
                 f'mean {values.mean():.3f}, median {np.median(values):.3f}, '
                 f'≥0.95: {sum(v >= 0.95 for v in values)}/61, '
                 f'≥0.90: {sum(v >= 0.90 for v in values)}/61',
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'sci_scoreboard.png', dpi=140,
                bbox_inches='tight')
    plt.close(fig)


def fig_jsr_curve():
    """k_eff/k_ff curves: symmetric J/(J+6), and asymmetric with one end fixed."""
    Js = np.linspace(0, 30, 300)
    sym = [_semirigid_factor(J, J) for J in Js]
    top_rigid = [_semirigid_factor(np.inf, J) for J in Js]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Js, sym,        lw=2.0, label=r'symmetric  $J_t = J_b = J$')
    ax.plot(Js, top_rigid,  lw=2.0, label=r'asymmetric  $J_t \to \infty$')
    ax.axhline(1.0, color='black', lw=0.6, ls=':')
    ax.axhline(0.5, color='gray', lw=0.6, ls=':')
    ax.axhline(0.25, color='gray', lw=0.6, ls=':')
    ax.axvline(6.0,  color='gray', lw=0.6, ls=':')
    ax.text(6.5, 0.52, '$J=6$\n$k_{eff}/k_{ff} = 0.5$', fontsize=8, color='gray')
    ax.text(0.3, 0.27, 'top-rigid limit:\n$k_{eff}/k_{ff} \\to 1/4$',
            fontsize=8, color='gray')
    ax.set_xlabel(r'Joint stiffness ratio $J = k_r L / EI$')
    ax.set_ylabel(r'Effective lateral stiffness ratio $k_{eff} / k_{ff}$')
    ax.set_xlim(0, 30); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_title('Semi-rigid joint correction factor', fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'jsr_curve.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_jsr_heatmap():
    """Heatmap of k_eff/k_ff over (J_t, J_b)."""
    Js = np.linspace(0.1, 20, 80)
    Z = np.zeros((len(Js), len(Js)))
    for i, jt in enumerate(Js):
        for j, jb in enumerate(Js):
            Z[i, j] = _semirigid_factor(jt, jb)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Z, origin='lower', extent=[Js[0], Js[-1], Js[0], Js[-1]],
                   cmap='viridis', vmin=0, vmax=1, aspect='equal')
    cs = ax.contour(Js, Js, Z, levels=[0.1, 0.25, 0.5, 0.75, 0.9],
                    colors='white', linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    ax.set_xlabel(r'$J_b$  (bottom-end joint stiffness ratio)')
    ax.set_ylabel(r'$J_t$  (top-end joint stiffness ratio)')
    ax.set_title(r'Asymmetric semi-rigid factor  $k_{eff}/k_{ff}(J_t, J_b)$',
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$k_{eff}/k_{ff}$')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'asymmetric_jsr.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    print('Generating figures into', OUT_DIR)
    fig_building_pristine();    print('  ✓ building_pristine.png')
    fig_building_combined();    print('  ✓ building_d85_combined.png')
    fig_jsr_curve();            print('  ✓ jsr_curve.png')
    fig_jsr_heatmap();          print('  ✓ asymmetric_jsr.png')
    fig_frf('Pristine', 'frf_pristine.png');                     print('  ✓ frf_pristine.png')
    fig_frf('D(85%) 1AD + D(85%) 1BD', 'frf_d85_1ad_1bd.png');   print('  ✓ frf_d85_1ad_1bd.png')
    fig_frf('Mass First Floor', 'frf_mass_1f.png');              print('  ✓ frf_mass_1f.png')
    fig_cfdac('Pristine', 'cfdac_pristine.png');                 print('  ✓ cfdac_pristine.png')
    fig_cfdac('D(85%) 2BD', 'cfdac_d85_2bd.png');                print('  ✓ cfdac_d85_2bd.png')
    fig_cfdac('D(85%) 1AD + D(85%) 1BD', 'cfdac_d85_1ad_1bd.png'); print('  ✓ cfdac_d85_1ad_1bd.png')
    fig_sci_scoreboard();       print('  ✓ sci_scoreboard.png')


if __name__ == '__main__':
    main()
