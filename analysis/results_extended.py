#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN Paralelo: Análisis Extendido de Métricas
=============================================
Genera figuras adicionales más allá del speedup:
  fig_6:  Tiempo absoluto vs N (escala log-log)
  fig_7:  Eficiencia OpenMP vs hilos (múltiples N)
  fig_8:  Tiempo por query vs N
  fig_9:  Overhead de transferencia CUDA (% y absoluto)
  fig_10: Impacto de K en el rendimiento
  fig_11: Coeficiente de variación (estabilidad)
  fig_12: Ley de Amdahl — ajuste teórico vs real
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import sys
from scipy.optimize import curve_fit

# ── Config ──────────────────────────────────────────────────────────
SOFT_MODE = '--soft' in sys.argv

sns.set_theme(style='whitegrid', palette='viridis')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
})

FIGS_DIR = 'analysis/soft' if SOFT_MODE else 'analysis/figures'
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────
CSV_FILE = ('results/benchmark_results_soft.csv' if SOFT_MODE
            else 'results/benchmark_results.csv')
print(f"Cargando: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

# Derive Q from the data pattern (Q = N/5, min 100)
df['Q'] = df['n'].apply(lambda n: max(n // 5, 100))

agg = df.groupby(['n', 'd', 'k', 'impl', 'threads']).agg(
    time_ms_mean=('time_ms', 'mean'),
    time_ms_std=('time_ms', 'std'),
    transfer_ms_mean=('transfer_ms', 'mean'),
    compute_ms_mean=('compute_ms', 'mean'),
    speedup_mean=('speedup_vs_seq', 'mean'),
    speedup_std=('speedup_vs_seq', 'std'),
    Q=('Q', 'first'),
).reset_index()

# Seq reference times
seq_ref = (agg[agg['impl'] == 'seq']
           [['n', 'd', 'k', 'time_ms_mean']]
           .rename(columns={'time_ms_mean': 'seq_time'}))
agg = agg.merge(seq_ref, on=['n', 'd', 'k'], how='left')

# Per-query time
agg['time_per_query_ms'] = agg['time_ms_mean'] / agg['Q']

# Efficiency for OMP
omp_mask = agg['impl'] == 'omp'
agg.loc[omp_mask, 'efficiency'] = (
    agg.loc[omp_mask, 'speedup_mean'] / agg.loc[omp_mask, 'threads'] * 100
)

# CUDA transfer overhead
cuda_mask = agg['impl'] == 'cuda'
agg.loc[cuda_mask, 'transfer_pct'] = (
    agg.loc[cuda_mask, 'transfer_ms_mean'] / agg.loc[cuda_mask, 'time_ms_mean'] * 100
)
agg.loc[cuda_mask, 'speedup_compute'] = (
    agg.loc[cuda_mask, 'seq_time'] / agg.loc[cuda_mask, 'compute_ms_mean']
)

# CV (coefficient of variation) per run group
cv_data = df.groupby(['n', 'd', 'k', 'impl', 'threads']).agg(
    cv=('time_ms', lambda x: x.std() / x.mean() * 100 if x.mean() > 0 else 0)
).reset_index()

# Adaptive params
N_MAX = agg['n'].max()
D_REF = 100 if 100 in agg['d'].unique() else agg['d'].max()
K_REF = 5 if 5 in agg['k'].unique() else sorted(agg['k'].unique())[-1]
N_VALS = sorted(agg['n'].unique())
D_VALS = sorted(agg['d'].unique())

print(f"N values: {N_VALS}, D values: {D_VALS}")
print(f"Reference: D={D_REF}, K={K_REF}, N_MAX={N_MAX}")

# ── Color palette ───────────────────────────────────────────────────
COLORS_OMP = sns.color_palette('Blues', 7)[2:]
COLOR_CUDA = '#e74c3c'
COLOR_CUDA_COMPUTE = '#2ecc71'
COLOR_SEQ = '#7f8c8d'


# ════════════════════════════════════════════════════════════════════
# Fig 6: Tiempo absoluto vs N (log-log)
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))
mask = (agg['d'] == D_REF) & (agg['k'] == K_REF)
data = agg[mask].copy()

# Sequential
sub = data[data['impl'] == 'seq'].sort_values('n')
if len(sub) > 0:
    ax.plot(sub['n'], sub['time_ms_mean'], 'o-', color=COLOR_SEQ,
            label='Secuencial', linewidth=2, markersize=6)

# OMP best (8 threads)
threads_list = sorted(data[data['impl'] == 'omp']['threads'].unique())
best_t = 8 if 8 in threads_list else threads_list[-1] if threads_list else 1
sub = data[(data['impl'] == 'omp') & (data['threads'] == best_t)].sort_values('n')
if len(sub) > 0:
    ax.plot(sub['n'], sub['time_ms_mean'], 's-', color=COLORS_OMP[2],
            label=f'OMP (T={int(best_t)})', linewidth=2, markersize=6)

# CUDA total & compute
sub = data[data['impl'] == 'cuda'].sort_values('n')
if len(sub) > 0:
    ax.plot(sub['n'], sub['time_ms_mean'], 'D-', color=COLOR_CUDA,
            label='CUDA (total)', linewidth=2, markersize=6)
    ax.plot(sub['n'], sub['compute_ms_mean'], '^--', color=COLOR_CUDA_COMPUTE,
            label='CUDA (solo cómputo)', linewidth=2, markersize=6)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('N (puntos de entrenamiento)')
ax.set_ylabel('Tiempo (ms)')
ax.set_title(f'Tiempo Absoluto de Ejecución vs N  (D={int(D_REF)}, K={int(K_REF)})')
ax.legend(loc='upper left')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_6.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  ✓ fig_6.png — Tiempo absoluto vs N")


# ════════════════════════════════════════════════════════════════════
# Fig 7: Eficiencia OpenMP vs hilos (múltiples N)
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
mask = (agg['d'] == D_REF) & (agg['k'] == K_REF) & (agg['impl'] == 'omp')
data = agg[mask].copy()

palette = sns.color_palette('viridis', len(N_VALS))
for n_val, color in zip(N_VALS, palette):
    sub = data[data['n'] == n_val].sort_values('threads')
    if len(sub) > 0 and 'efficiency' in sub.columns:
        ax.plot(sub['threads'], sub['efficiency'], 'o-', color=color,
                label=f'N={n_val:,}', linewidth=2, markersize=6)

ax.axhline(y=100, color='gray', linestyle='--', alpha=0.6, label='Ideal (100%)')
ax.set_xlabel('Número de hilos')
ax.set_ylabel('Eficiencia (%)')
ax.set_title(f'Eficiencia OpenMP = Speedup/Hilos  (D={int(D_REF)}, K={int(K_REF)})')
ax.set_xticks(sorted(data['threads'].unique()) if len(data) > 0 else [])
ax.set_ylim(0, 120)
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_7.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  ✓ fig_7.png — Eficiencia OpenMP")


# ════════════════════════════════════════════════════════════════════
# Fig 8: Tiempo por query vs N
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))
mask = (agg['d'] == D_REF) & (agg['k'] == K_REF)
data = agg[mask].copy()

sub = data[data['impl'] == 'seq'].sort_values('n')
if len(sub) > 0:
    ax.plot(sub['n'], sub['time_per_query_ms'], 'o-', color=COLOR_SEQ,
            label='Secuencial', linewidth=2, markersize=6)

sub = data[(data['impl'] == 'omp') & (data['threads'] == best_t)].sort_values('n')
if len(sub) > 0:
    ax.plot(sub['n'], sub['time_per_query_ms'], 's-', color=COLORS_OMP[2],
            label=f'OMP (T={int(best_t)})', linewidth=2, markersize=6)

sub = data[data['impl'] == 'cuda'].sort_values('n')
if len(sub) > 0:
    ax.plot(sub['n'], sub['time_per_query_ms'], 'D-', color=COLOR_CUDA,
            label='CUDA (total)', linewidth=2, markersize=6)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('N (puntos de entrenamiento)')
ax.set_ylabel('Tiempo por query (ms/query)')
ax.set_title(f'Latencia por Query vs N  (D={int(D_REF)}, K={int(K_REF)})')
ax.legend(loc='upper left')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_8.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  ✓ fig_8.png — Tiempo por query")


# ════════════════════════════════════════════════════════════════════
# Fig 9: Overhead de transferencia CUDA
# ════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

cuda_agg = agg[agg['impl'] == 'cuda'].copy()

# Left: Transfer % vs N for different D
for d_val, color in zip(D_VALS, sns.color_palette('plasma', len(D_VALS))):
    sub = cuda_agg[(cuda_agg['d'] == d_val) & (cuda_agg['k'] == K_REF)].sort_values('n')
    if len(sub) > 0:
        ax1.plot(sub['n'], sub['transfer_pct'], 'o-', color=color,
                 label=f'D={d_val}', linewidth=2, markersize=5)

ax1.set_xscale('log')
ax1.set_xlabel('N')
ax1.set_ylabel('Transferencia H↔D (%)')
ax1.set_title(f'Overhead de Transferencia CUDA (K={int(K_REF)})')
ax1.legend(fontsize=7, ncol=2)

# Right: Stacked - transfer vs compute for biggest N, varying D
sub = cuda_agg[(cuda_agg['n'] == N_MAX) & (cuda_agg['k'] == K_REF)].sort_values('d')
if len(sub) > 0:
    x = np.arange(len(sub))
    width = 0.6
    ax2.bar(x, sub['compute_ms_mean'].values, width, label='Cómputo', color='#3498db')
    ax2.bar(x, sub['transfer_ms_mean'].values, width,
            bottom=sub['compute_ms_mean'].values, label='Transfer H↔D', color='#e74c3c')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(d)) for d in sub['d'].values])
    ax2.set_xlabel('D (características)')
    ax2.set_ylabel('Tiempo (ms)')
    ax2.set_title(f'CUDA: Desglose por D  (N={N_MAX:,}, K={int(K_REF)})')
    ax2.legend()

fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_9.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  ✓ fig_9.png — Overhead transferencia CUDA")


# ════════════════════════════════════════════════════════════════════
# Fig 10: Impacto de K en el rendimiento
# ════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

k_vals = sorted(agg['k'].unique())

# Left: Time vs K for each impl at largest N, D=D_REF
mask_base = (agg['n'] == N_MAX) & (agg['d'] == D_REF)

sub_seq = agg[mask_base & (agg['impl'] == 'seq')].sort_values('k')
if len(sub_seq) > 0:
    ax1.plot(sub_seq['k'], sub_seq['time_ms_mean'], 'o-', color=COLOR_SEQ,
             label='Secuencial', linewidth=2, markersize=6)

sub_omp = agg[mask_base & (agg['impl'] == 'omp') & (agg['threads'] == best_t)].sort_values('k')
if len(sub_omp) > 0:
    ax1.plot(sub_omp['k'], sub_omp['time_ms_mean'], 's-', color=COLORS_OMP[2],
             label=f'OMP (T={int(best_t)})', linewidth=2, markersize=6)

sub_cuda = agg[mask_base & (agg['impl'] == 'cuda')].sort_values('k')
if len(sub_cuda) > 0:
    ax1.plot(sub_cuda['k'], sub_cuda['time_ms_mean'], 'D-', color=COLOR_CUDA,
             label='CUDA (total)', linewidth=2, markersize=6)

ax1.set_xlabel('K (vecinos)')
ax1.set_ylabel('Tiempo (ms)')
ax1.set_title(f'Efecto de K en el Tiempo  (N={N_MAX:,}, D={int(D_REF)})')
ax1.set_xticks(k_vals)
ax1.legend()

# Right: Speedup vs K
if len(sub_omp) > 0:
    ax2.plot(sub_omp['k'], sub_omp['speedup_mean'], 's-', color=COLORS_OMP[2],
             label=f'OMP (T={int(best_t)})', linewidth=2, markersize=6)
if len(sub_cuda) > 0:
    ax2.plot(sub_cuda['k'], sub_cuda['speedup_mean'], 'D-', color=COLOR_CUDA,
             label='CUDA (total)', linewidth=2, markersize=6)
    if 'speedup_compute' in sub_cuda.columns:
        ax2.plot(sub_cuda['k'], sub_cuda['speedup_compute'], '^--',
                 color=COLOR_CUDA_COMPUTE, label='CUDA (solo cómputo)',
                 linewidth=2, markersize=6)

ax2.set_xlabel('K (vecinos)')
ax2.set_ylabel('Speedup vs Secuencial')
ax2.set_title(f'Efecto de K en el Speedup  (N={N_MAX:,}, D={int(D_REF)})')
ax2.set_xticks(k_vals)
ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax2.legend()

fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_10.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  ✓ fig_10.png — Impacto de K")


# ════════════════════════════════════════════════════════════════════
# Fig 11: Coeficiente de variación (estabilidad)
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

cv_summary = cv_data.groupby('impl')['cv'].agg(['mean', 'max', 'median']).reset_index()

# Box plot of CV by implementation
impls_order = ['seq', 'omp', 'cuda']
impl_colors = {'seq': COLOR_SEQ, 'omp': COLORS_OMP[2], 'cuda': COLOR_CUDA}

box_data = []
box_labels = []
box_colors = []
for impl in impls_order:
    vals = cv_data[cv_data['impl'] == impl]['cv'].values
    if len(vals) > 0:
        box_data.append(vals)
        box_labels.append(impl.upper())
        box_colors.append(impl_colors.get(impl, 'gray'))

if box_data:
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

ax.set_ylabel('Coeficiente de Variación (%)')
ax.set_title('Estabilidad de Mediciones: CV por Implementación')
ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='CV=5% (umbral bueno)')
ax.legend()
fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_11.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  ✓ fig_11.png — Coeficiente de variación")


# ════════════════════════════════════════════════════════════════════
# Fig 12: Ley de Amdahl — ajuste teórico vs real
# ════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))

mask = (agg['n'] == N_MAX) & (agg['d'] == D_REF) & (agg['k'] == K_REF) & (agg['impl'] == 'omp')
omp_scale = agg[mask].sort_values('threads')

if len(omp_scale) > 0:
    threads = omp_scale['threads'].values
    speedup = omp_scale['speedup_mean'].values

    # Amdahl's law: S(p) = 1 / (f + (1-f)/p)
    def amdahl(p, f):
        return 1.0 / (f + (1.0 - f) / p)

    try:
        popt, _ = curve_fit(amdahl, threads, speedup, p0=[0.05], bounds=(0, 1))
        f_serial = popt[0]
        p_fit = np.linspace(1, max(threads) * 1.5, 100)
        s_fit = amdahl(p_fit, f_serial)

        ax.plot(threads, speedup, 'o', color=COLORS_OMP[2], markersize=8,
                label='Speedup medido', zorder=5)
        ax.plot(p_fit, s_fit, '-', color=COLORS_OMP[3], linewidth=2,
                label=f'Amdahl (f={f_serial:.4f})')
        ax.plot(p_fit, p_fit, '--', color='gray', alpha=0.5, label='Ideal (S=p)')

        # Max theoretical speedup
        s_max = 1.0 / f_serial if f_serial > 0 else float('inf')
        ax.axhline(y=s_max, color='red', linestyle=':', alpha=0.6,
                   label=f'Límite Amdahl = {s_max:.1f}x')

        ax.set_xlabel('Número de hilos (p)')
        ax.set_ylabel('Speedup')
        ax.set_title(f'Ley de Amdahl: OMP  (N={N_MAX:,}, D={int(D_REF)}, K={int(K_REF)})\n'
                     f'Fracción serial estimada: f = {f_serial:.4f}')
        ax.legend(loc='upper left')
        ax.set_xticks(threads)
    except Exception as e:
        ax.text(0.5, 0.5, f'Error en ajuste: {e}', ha='center',
                transform=ax.transAxes)
else:
    ax.text(0.5, 0.5, 'Sin datos OMP disponibles', ha='center',
            transform=ax.transAxes)

fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_12.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("  ✓ fig_12.png — Ley de Amdahl")


# ── Summary ─────────────────────────────────────────────────────────
print(f"\nFiguras extendidas guardadas en {FIGS_DIR}/")
for f in sorted(os.listdir(FIGS_DIR)):
    if f.endswith('.png'):
        size_kb = os.path.getsize(f'{FIGS_DIR}/{f}') / 1024
        print(f"  {f}  ({size_kb:.0f} KB)")
