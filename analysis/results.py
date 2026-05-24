# %% [markdown]
# # KNN Paralelo: OpenMP vs CUDA — Análisis de Resultados
#
# Este notebook procesa `results/benchmark_results.csv` y genera las 5
# figuras del informe: speedup vs N, speedup vs D, heatmap CUDA/OpenMP,
# desglose de tiempo CUDA y escalabilidad fuerte OpenMP.

# %%
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import sys

SOFT_MODE = '--soft' in sys.argv

sns.set_theme(style='whitegrid', palette='viridis')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

FIGS_DIR = 'analysis/figures'
os.makedirs(FIGS_DIR, exist_ok=True)

# %%
CSV_FILE = 'results/benchmark_results_soft.csv' if SOFT_MODE else 'results/benchmark_results.csv'
print(f"Cargando: {CSV_FILE} ({'soft' if SOFT_MODE else 'full'} mode)")
df = pd.read_csv(CSV_FILE)

agg = df.groupby(['n', 'd', 'k', 'impl', 'threads']).agg(
    time_ms_mean=('time_ms', 'mean'),
    time_ms_std=('time_ms', 'std'),
    transfer_ms_mean=('transfer_ms', 'mean'),
    compute_ms_mean=('compute_ms', 'mean'),
    speedup_mean=('speedup_vs_seq', 'mean'),
    speedup_std=('speedup_vs_seq', 'std'),
).reset_index()

seq_ref = (agg[agg['impl'] == 'seq']
           [['n', 'd', 'k', 'time_ms_mean']]
           .rename(columns={'time_ms_mean': 'seq_time'}))
agg = agg.merge(seq_ref, on=['n', 'd', 'k'], how='left')

cuda_mask = agg['impl'] == 'cuda'
agg.loc[cuda_mask, 'speedup_compute'] = (
    agg.loc[cuda_mask, 'seq_time'] / agg.loc[cuda_mask, 'compute_ms_mean']
)
agg.loc[~cuda_mask, 'speedup_compute'] = np.nan

print(f"Datos cargados: {len(df)} filas, {len(agg)} agregados")
print(f"Implementations: {sorted(agg['impl'].unique())}")
print(f"N range: {agg['n'].min()} - {agg['n'].max()}")
print(f"D range: {agg['d'].min()} - {agg['d'].max()}")
print(f"K range: {sorted(agg['k'].unique())}")

# Adaptive figure parameters
_FIG1_D = 100 if 100 in agg['d'].unique() else agg['d'].max()
_FIG1_K = 5 if 5 in agg['k'].unique() else agg['k'].iloc[0]
_FIG2_N = 50000 if 50000 in agg['n'].unique() else agg['n'].max()
_FIG2_K = 5 if 5 in agg['k'].unique() else agg['k'].iloc[0]
_FIG4_PIE_N = 100000 if 100000 in agg['n'].unique() else agg['n'].max()
_FIG4_PIE_D = 500 if 500 in agg['d'].unique() else agg['d'].max()
_FIG4_PIE_K = 5 if 5 in agg['k'].unique() else agg['k'].iloc[0]
_FIG5_N = 100000 if 100000 in agg['n'].unique() else agg['n'].max()
_FIG5_D = 100 if 100 in agg['d'].unique() else agg['d'].max()
_FIG5_K = 5 if 5 in agg['k'].unique() else agg['k'].iloc[0]
print(f"Fig params: D1={_FIG1_D} N2={_FIG2_N} N4={_FIG4_PIE_N} D4={_FIG4_PIE_D} N5={_FIG5_N} D5={_FIG5_D}")
print(f"K range: {sorted(agg['k'].unique())}")

# %% [markdown]
# ## 1. Speedup vs N (D=_FIG1_D, K=_FIG1_K)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

mask = (agg['d'] == _FIG1_D) & (agg['k'] == _FIG1_K) & (agg['impl'] != 'seq')
data = agg[mask].copy()

colors = sns.color_palette('viridis', 6)
linestyles = {'omp': '--', 'cuda': '-'}

lines = []
# OMP lines
for t, color in zip(sorted(data[data['impl'] == 'omp']['threads'].unique()), colors[:5]):
    sub = data[(data['impl'] == 'omp') & (data['threads'] == t)].sort_values('n')
    if len(sub) > 0:
        l, = ax.plot(sub['n'], sub['speedup_mean'], 'o-', color=color,
                      label=f'OMP threads={int(t)}', markersize=5)
        lines.append(l)

# CUDA lines
cuda_data = data[data['impl'] == 'cuda'].sort_values('n')
if len(cuda_data) > 0:
    l1, = ax.plot(cuda_data['n'], cuda_data['speedup_compute'], 's-', color=colors[4],
                   label='CUDA (solo compute)', markersize=5, linewidth=2)
    l2, = ax.plot(cuda_data['n'], cuda_data['speedup_mean'], 'D-', color='red',
                   label='CUDA (total)', markersize=5, linewidth=2)
    lines.extend([l1, l2])

ax.set_xscale('log')
ax.set_xlabel('N (puntos de entrenamiento)')
ax.set_ylabel('Speedup vs Secuencial')
ax.set_title(f'Speedup vs N  (D={int(_FIG1_D)}, K={int(_FIG1_K)})')
if lines:
    ax.legend(fontsize=8, loc='upper left')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_1.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 2. Speedup vs D (N=_FIG2_N, K=_FIG2_K)

# %%
fig, ax = plt.subplots(figsize=(8, 5))

mask = (agg['n'] == _FIG2_N) & (agg['k'] == _FIG2_K) & (agg['impl'] != 'seq')
data = agg[mask].copy()

lines = []
# OMP lines
for t, color in zip(sorted(data[data['impl'] == 'omp']['threads'].unique()), colors[:5]):
    sub = data[(data['impl'] == 'omp') & (data['threads'] == t)].sort_values('d')
    if len(sub) > 0:
        l, = ax.plot(sub['d'], sub['speedup_mean'], 'o-', color=color,
                      label=f'OMP threads={int(t)}', markersize=5)
        lines.append(l)

# CUDA lines
cuda_data = data[data['impl'] == 'cuda'].sort_values('d')
if len(cuda_data) > 0:
    l1, = ax.plot(cuda_data['d'], cuda_data['speedup_compute'], 's-', color=colors[4],
                   label='CUDA (solo compute)', markersize=5, linewidth=2)
    l2, = ax.plot(cuda_data['d'], cuda_data['speedup_mean'], 'D-', color='red',
                   label='CUDA (total)', markersize=5, linewidth=2)
    lines.extend([l1, l2])

ax.set_xscale('log')
ax.set_xlabel('D (caracteristicas)')
ax.set_ylabel('Speedup vs Secuencial')
ax.set_title(f'Speedup vs D  (N={int(_FIG2_N)}, K={int(_FIG2_K)})')
if lines:
    ax.legend(fontsize=8, loc='upper left')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_2.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Heatmap: Speedup CUDA vs Mejor OpenMP (K=5)

# %%
fig, ax = plt.subplots(figsize=(10, 6))

k5 = agg[(agg['k'] == 5) & (agg['impl'] != 'seq')].copy()

omp_best = (k5[k5['impl'] == 'omp']
            .groupby(['n', 'd'])['speedup_mean']
            .max()
            .reset_index(name='omp_best_speedup'))

cuda_sp = (k5[k5['impl'] == 'cuda']
           [['n', 'd', 'speedup_mean']]
           .rename(columns={'speedup_mean': 'cuda_speedup'}))

heat = omp_best.merge(cuda_sp, on=['n', 'd'], how='inner')
heat['ratio'] = heat['cuda_speedup'] / heat['omp_best_speedup']
heat['ratio'] = heat['ratio'].clip(0, 3)

if len(heat) > 0:
    pivot = heat.pivot_table(index='n', columns='d', values='ratio', aggfunc='mean')
    pivot = pivot.sort_index(ascending=True)
    if not pivot.empty:
        sns.heatmap(pivot, ax=ax, cmap='RdYlGn', center=1.0, annot=True, fmt='.2f',
                    linewidths=0.5, cbar_kws={'label': 'Speedup CUDA / Speedup Mejor OMP'})
ax.set_title('Heatmap: CUDA vs Mejor OpenMP  (K=5)')
ax.set_xlabel('D (caracteristicas)')
ax.set_ylabel('N (puntos de entrenamiento)')
if len(heat) == 0:
    ax.text(0.5, 0.5, 'No CUDA data for heatmap', ha='center',
            va='center', transform=ax.transAxes)

fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_3.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Desglose de Tiempo CUDA

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

cuda_all = df[df['impl'] == 'cuda'].copy()

# Pie chart with adaptive parameters
pie_mask = (cuda_all['n'] == _FIG4_PIE_N) & (cuda_all['d'] == _FIG4_PIE_D) & (cuda_all['k'] == _FIG4_PIE_K)
pie_data = cuda_all[pie_mask]
if len(pie_data) > 0:
    tfr_mean = pie_data['transfer_ms'].mean()
    cmp_mean = pie_data['compute_ms'].mean()
    sizes = [tfr_mean, cmp_mean]
    labels = [f'Transfer H↔D\n({tfr_mean:.1f} ms)', f'Compute\n({cmp_mean:.1f} ms)']
    colors_pie = ['#ff9999', '#66b3ff']
    ax1.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, explode=(0.02, 0))
    ax1.set_title(f'CUDA: N={int(_FIG4_PIE_N)} D={int(_FIG4_PIE_D)} K={int(_FIG4_PIE_K)}')
else:
    # Fallback: try any available config
    cuda_agg = (cuda_all.groupby(['n', 'd', 'k'])
                [['transfer_ms', 'compute_ms']].mean().reset_index())
    if len(cuda_agg) > 0:
        row = cuda_agg.iloc[0]
        sizes = [row['transfer_ms'], row['compute_ms']]
        labels = [f'Transfer H↔D\n({row["transfer_ms"]:.1f} ms)',
                  f'Compute\n({row["compute_ms"]:.1f} ms)']
        ax1.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                startangle=90, explode=(0.02, 0))
        ax1.set_title(f'CUDA: N={int(row["n"])} D={int(row["d"])} K={int(row["k"])}')
    else:
        ax1.text(0.5, 0.5, 'No CUDA data', ha='center', transform=ax1.transAxes)

# Stacked bars: transfer vs compute, adaptive K
bar_data = (cuda_all[cuda_all['k'] == _FIG4_PIE_K]
            .groupby('n')[['transfer_ms', 'compute_ms']]
            .mean()
            .sort_index())

if len(bar_data) > 0:
    x = np.arange(len(bar_data))
    width = 0.6
    ax2.bar(x, bar_data['compute_ms'], width, label='Compute', color='#66b3ff')
    ax2.bar(x, bar_data['transfer_ms'], width, bottom=bar_data['compute_ms'],
            label='Transfer H↔D', color='#ff9999')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(n)) for n in bar_data.index], rotation=45)
    ax2.set_xlabel('N')
    ax2.set_ylabel('Tiempo (ms)')
    ax2.set_title('CUDA: Transfer vs Compute por N (K=5)')
    ax2.legend(fontsize=8)
else:
    ax2.text(0.5, 0.5, 'No CUDA data', ha='center', transform=ax2.transAxes)

fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_4.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Escalabilidad Fuerte OpenMP

# %%
fig, ax1 = plt.subplots(figsize=(7, 5))

mask = (agg['n'] == _FIG5_N) & (agg['d'] == _FIG5_D) & (agg['k'] == _FIG5_K) & (agg['impl'] == 'omp')
scale_data = agg[mask].sort_values('threads')

if len(scale_data) == 0:
    mask = (agg['d'] == _FIG5_D) & (agg['k'] == _FIG5_K) & (agg['impl'] == 'omp')
    scale_data = agg[mask].sort_values(['n', 'threads'])
    if len(scale_data) > 0:
        best_n = scale_data['n'].max()
        scale_data = scale_data[scale_data['n'] == best_n].sort_values('threads')

if len(scale_data) > 0:
    threads = scale_data['threads'].values
    speedup = scale_data['speedup_mean'].values
    base_speedup = speedup[threads == 1][0] if 1 in threads else speedup[0]
    rel_speedup = speedup / base_speedup
    efficiency = rel_speedup / threads * 100

    color1 = '#2c7bb6'
    ax1.plot(threads, rel_speedup, 'o-', color=color1, linewidth=2,
             markersize=8, label='Speedup real')
    ax1.plot(threads, threads, '--', color='gray', alpha=0.7, label='Ideal (S=p)')
    ax1.set_xlabel('Numero de hilos (p)')
    ax1.set_ylabel('Speedup relativo', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(threads)

    ax2 = ax1.twinx()
    color2 = '#d7191c'
    ax2.plot(threads, efficiency, 's--', color=color2, linewidth=2,
             markersize=7, label='Eficiencia (%)')
    ax2.set_ylabel('Eficiencia S/p (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    max_n = scale_data['n'].max() if len(scale_data) > 0 else 'N/A'
    ax1.set_title(f'Escalabilidad Fuerte OpenMP  (N={int(max_n)}, D={int(scale_data["d"].max())}, K={int(scale_data["k"].max())})')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
else:
    ax1.text(0.5, 0.5, 'No OpenMP data available', ha='center', transform=ax1.transAxes)

fig.tight_layout()
fig.savefig(f'{FIGS_DIR}/fig_5.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
print(f"\nFiguras guardadas en {FIGS_DIR}/")
for f in sorted(os.listdir(FIGS_DIR)):
    if f.endswith('.png'):
        size_kb = os.path.getsize(f'{FIGS_DIR}/{f}') / 1024
        print(f"  {f}  ({size_kb:.0f} KB)")
