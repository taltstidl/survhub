"""
Plots for publication
"""
from collections import namedtuple
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bencheval.bencheval.tabarena import TabArena
from bencheval.bencheval.winrate_utils import compute_winrate_matrix
from utils import style_boxplot

# Basic style parameters
plt.rcParams['font.family'] = 'FAUSans Office'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 8

# Color definitions
green, blue, purple = '#00BD9D', '#00BBF9', '#9B5DE5'
colors = [(0.0, '#9B5DE5'), (0.46, '#00BBF9'), (1.0, '#00BD9D')]
cmap = LinearSegmentedColormap.from_list('kdd', colors)
colors = [(0.0, '#9B5DE5'), (0.5, 'white'), (1.0, '#00BD9D')]
cmap_diverging = LinearSegmentedColormap.from_list('kdd_diverging', colors)

# Model-specific style definitions
Style = namedtuple('Style', ['name', 'color'])
styles: dict[str, Style] = {
    'coxnet': Style('Coxnet', green),
    'coxnet-tuned': Style('Coxnet (Tuned)', green),
    'rsf': Style('RSF', green),
    'rsf-tuned': Style('RSF (Tuned) ', green),
    'gbse': Style('GBSE', green),
    'gbse-tuned': Style('GBSE (Tuned) ', green),
    'ssvm': Style('SSVM', green),
    'ssvm-tuned': Style('SSVM (Tuned)', green),
    'deepsurv': Style('DeepSurv', blue),
    'deephit': Style('DeepHit', blue),
    'deepweisurv1': Style('DeepWeiSurv (p=1)', blue),
    'deepweisurv2': Style('DeepWeiSurv (p=2)', blue),
    'rankdeepsurv': Style('RankDeepSurv', blue),
    'tabpfn': Style('TabPFN$^*$', purple),
    'popsicl': Style('PopSICL', purple),
    'none': Style('None', '#eee'),
}


def plot_datasets():
    df_datasets = pd.read_csv(Path('summary.csv'))
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    scatter = ax.scatter(df_datasets['n_samples'], df_datasets['n_num_cols'] + df_datasets['n_cat_cols'],
                         c=100 * df_datasets['censoring_ratio'], s=5, cmap=cmap)
    ax.set_xlabel('Number of Samples')
    ax.set_xscale('log')
    ax.set_ylabel('Number of Features')
    ax.set_yscale('log')
    ax.spines[['top', 'right']].set_visible(False)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Censoring Ratio')
    cbar.outline.set_visible(False)
    plt.tight_layout()
    plt.savefig('kdd_datasets.pdf', bbox_inches='tight')


def load_leaderboard(subset: Literal['small', 'large'] = 'small'):
    df_results = pd.read_csv(f'benchmark/results_{subset}.csv')
    df_results['error'] = 1.0 - df_results['c_index']  # leaderboard needs errors, i.e., smaller is better
    tabarena = TabArena(method_col='model', task_col='dataset', seed_column='fold', error_col='error',
                        columns_to_agg_extra=['fit_time', 'predict_time'])
    elo_settings = {'calibration_framework': 'rsf', 'calibration_elo': 1000, 'BOOTSTRAP_ROUNDS': 100}
    leaderboard = tabarena.leaderboard(data=df_results, include_winrate=True, include_mrr=True,
                                       include_rank_counts=True, include_elo=True, elo_kwargs=elo_settings)
    leaderboard.to_csv(f'leaderboard_{subset}.csv')
    return leaderboard


def plot_leaderboard():
    fig, axes = plt.subplots(figsize=(7, 2.5), ncols=2, width_ratios=[20.5, 14.5], sharey=True)

    # Leaderboard for small datasets
    leaderboard = load_leaderboard('small')
    #locs = [1, 2, 3.5, 4.5, 6, 7, 8.5, 10, 11.5, 12.5, 14, 15.5, 17]
    locs = [1, 2, 3.5, 4.5, 6, 7, 8.5, 9.5, 11, 12.5, 14, 15, 16.5, 18, 19.5]
    keys = ['coxnet', 'coxnet-tuned', 'rsf', 'rsf-tuned', 'gbse', 'gbse-tuned', 'ssvm', 'ssvm-tuned', 'deepsurv',
            'deephit', 'deepweisurv1', 'deepweisurv2', 'rankdeepsurv', 'tabpfn', 'popsicl']
    assert len(locs) == len(keys), 'Mismatch in bar definitions'
    scores = [0 if key == 'none' else leaderboard.loc[key]['elo'] for key in keys]
    scores_pos_err = [0 if key == 'none' else leaderboard.loc[key]['elo+'] for key in keys]
    scores_neg_err = [0 if key == 'none' else leaderboard.loc[key]['elo-'] for key in keys]
    ax = axes[0]
    cs = [styles[k].color for k in keys]
    ax.bar(locs, scores, yerr=[scores_neg_err, scores_pos_err], capsize=2, alpha=0.33, color=cs, edgecolor=cs,
           error_kw={'lw': 1, 'capthick': 1})
    ax.set_title('Small Datasets\nSamples ≤ 1000 & Features ≤ 100')
    ax.set_xlim(left=0, right=20.5)
    ax.set_xticks(locs, [styles[k].name for k in keys], rotation=35, ha='right')
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#eee')

    # Leaderboard for large datasets
    leaderboard = load_leaderboard('large')
    #locs = [1, 2, 3.5, 5, 6.5, 8, 9.5, 10.5, 12]
    locs = [1, 2, 3.5, 5, 6.5, 8, 9.5, 11, 12, 13.5]
    keys = ['coxnet', 'coxnet-tuned', 'rsf', 'gbse', 'ssvm', 'deepsurv', 'deephit', 'deepweisurv1', 'deepweisurv2',
            'rankdeepsurv']
    assert len(locs) == len(keys), 'Mismatch in bar definitions'
    scores = [0 if key == 'none' else leaderboard.loc[key]['elo'] for key in keys]
    scores_pos_err = [0 if key == 'none' else leaderboard.loc[key]['elo+'] for key in keys]
    scores_neg_err = [0 if key == 'none' else leaderboard.loc[key]['elo-'] for key in keys]
    ax = axes[1]
    cs = [styles[k].color for k in keys]
    ax.bar(locs, scores, yerr=[scores_neg_err, scores_pos_err], capsize=2, alpha=0.33, color=cs, edgecolor=cs,
           error_kw={'lw': 1, 'capthick': 1})
    ax.set_title('Large Datasets\nSamples > 1000 | Features > 100')
    ax.set_xlim(left=0, right=14.5)
    ax.set_xticks(locs, [styles[k].name for k in keys], rotation=35, ha='right')
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.tick_params(left=False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#eee')

    plt.tight_layout()
    plt.figtext(0.01, 0.01, '$^*$ Not a native survival model', ha='left', fontsize=7, color='#999')
    plt.savefig('kdd_leaderboard.pdf', bbox_inches='tight')
    plt.savefig('leaderboard.svg', bbox_inches='tight')


def plot_winrate_matrix(subset: Literal['small', 'large'] = 'small'):
    df_results = pd.read_csv(f'benchmark/results_{subset}.csv')
    df_results['error'] = 1.0 - df_results['c_index']  # leaderboard needs errors, i.e., smaller is better
    winrate_matrix = compute_winrate_matrix(df_results, method_col='model', task_col='dataset', seed_col='fold',
                                            error_col='error')
    winrate_matrix = (100 * winrate_matrix).round()

    fig, ax = plt.subplots(figsize=(3.35, 5))
    im = ax.imshow(winrate_matrix, cmap=cmap_diverging, aspect='equal')
    for i in range(len(winrate_matrix.index)):
        for j in range(len(winrate_matrix.columns)):
            winrate = winrate_matrix.iloc[i, j]
            if not np.isnan(winrate):
                color = 'white' if winrate > 75 or winrate < 25 else 'black'
                ax.text(j, i, f'{int(winrate)}', ha='center', va='center', color=color, fontsize=6)
    locations, labels = range(len(winrate_matrix.index)), [styles[k].name for k in winrate_matrix.index]
    ax.set_xticks(locations, labels, rotation=45, ha='right')
    ax.set_xlabel('Loser')
    ax.set_yticks(locations, labels, rotation=45, va='top')
    ax.set_ylabel('Winner')
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='4%', pad=0.1)
    plt.colorbar(im, cax=cax, orientation='horizontal', label='Win Rate [%]')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    # Get rid of axis splines
    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in cax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f'kdd_winrate_matrix_{subset}.pdf', bbox_inches='tight')


def plot_timing(subset: Literal['small', 'large'] = 'small'):
    df_results = pd.read_csv(f'benchmark/results_{subset}.csv')
    df_datasets = pd.read_csv('summary.csv')
    df = df_results.groupby(['model', 'dataset'])[['fit_time', 'predict_time']].sum().reset_index()
    df = pd.merge(df, df_datasets, left_on='dataset', right_on='name', how='left')
    # Compute fit (or predict for TFM) time per 1K samples
    df['time'] = np.where(df['model'].isin(['tabpfn', 'popsicl']), df['predict_time'], df['fit_time'])
    df['time'] = df['time'] / (20 * df['n_samples']) * 1000
    df = df[['model', 'dataset', 'time']]

    fig, ax = plt.subplots(figsize=(3.35, 2.8))
    keys = ['coxnet', 'coxnet-tuned', 'rsf', 'rsf-tuned', 'gbse', 'gbse-tuned', 'ssvm', 'ssvm-tuned', 'deepsurv',
            'deephit', 'deepweisurv1', 'deepweisurv2', 'rankdeepsurv', 'tabpfn', 'popsicl']
    bp = ax.boxplot([df[df['model'] == k]['time'].to_numpy() for k in keys], patch_artist=True)
    style_boxplot(bp, [styles[k].color for k in keys])
    ax.set_xticks(range(1, len(keys) + 1), [styles[k].name for k in keys], rotation=35, ha='right')
    ax.set_xlabel('')
    ax.set_ylabel('Time [s/1K samples]')
    ax.set_yscale('log')
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#eee')
    plt.tight_layout()
    plt.savefig(f'kdd_timing_{subset}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    plot_datasets()
    plot_leaderboard()
    plot_winrate_matrix(subset='small')
    plot_winrate_matrix(subset='large')
    plot_timing(subset='small')
    plot_timing(subset='large')
