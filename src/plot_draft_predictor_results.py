"""
Generate all figures for the adversarial draft model results section.

Produces five PNG files:
  adv_fig1_blue.png     — Blue-side drafting strength
  adv_fig2_red.png      — Red-side drafting strength
  adv_fig3_hth.png      — Head-to-head balance
  adv_fig4_heatmap.png  — Cross-model matchup heatmap
  adv_fig5_favpct.png   — Percentage of games favoured by side

Requirements: matplotlib, numpy
  pip install matplotlib numpy

Usage:
  python generate_figures.py
  python generate_figures.py --outdir results/figures
"""
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Data — paste updated values here if you rerun the evaluation
# ------------------------------------------------------------------

BASELINE = 0.4659   # Random vs Random mean Blue WP

# Always-Blue baseline: LR model's expected score when compositions are
# evaluated assuming Blue always wins — Blue win rate in test set ~52.7%
ALWAYS_BLUE_WP = 0.5273   # always-blue baseline Blue win probability
ALWAYS_RED_WP  = 1 - ALWAYS_BLUE_WP   # = 0.4727

RESULTS = {
    # (mean_blue_wp, blue_favored_pct, std)
    'mlp_blue_vs_random':    (0.5451, 63.4, 0.141),
    'random_vs_mlp_red':     (0.4787, 43.4, 0.141),
    'mlp_vs_mlp':            (0.5213, 55.8, 0.141),

    'lstm_blue_vs_random':   (0.5428, 64.0, 0.134),
    'random_vs_lstm_red':    (0.4703, 41.6, 0.139),
    'lstm_vs_lstm':          (0.4941, 47.4, 0.152),

    'xgb_blue_vs_random':    (0.5660, 69.8, 0.130),
    'random_vs_xgb_red':     (0.4707, 41.2, 0.149),
    'xgb_vs_xgb':            (0.5188, 54.0, 0.147),

    'mlp_blue_vs_lstm_red':  (0.5388, 59.0, 0.150),
    'mlp_blue_vs_xgb_red':   (0.5157, 54.0, 0.143),
    'lstm_blue_vs_xgb_red':  (0.4881, 45.0, 0.145),
}

# Colour palette — one colour per model + grey for random
COLORS = {
    'random': '#b0b8c1',
    'mlp':    '#4e9af1',
    'lstm':   '#f4a261',
    'xgb':    '#2ec4b6',
}

MODEL_LABELS = ['Random\n(baseline)', 'Winner MLP', 'Winner LSTM', 'Winner XGBoost']
MODEL_KEYS   = ['random', 'mlp', 'lstm', 'xgb']
BAR_COLORS   = [COLORS[k] for k in MODEL_KEYS]


def _style(ax):
    """Apply consistent styling to an axes object."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def _label_bars(ax, bars, fmt='{:.3f}', offset=0.003):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                fmt.format(bar.get_height()),
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')


# ------------------------------------------------------------------
# Figure 1 — Blue-side drafting strength
# ------------------------------------------------------------------

def fig_blue_strength(outpath):
    labels = ['Always-Blue\nBaseline', 'Winner MLP', 'Winner LSTM', 'Winner XGBoost']
    colors = [COLORS['random'], COLORS['mlp'], COLORS['lstm'], COLORS['xgb']]
    blue_wps = [
        ALWAYS_BLUE_WP,
        RESULTS['mlp_blue_vs_random'][0],
        RESULTS['lstm_blue_vs_random'][0],
        RESULTS['xgb_blue_vs_random'][0],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, blue_wps, color=colors,
                  edgecolor='white', linewidth=0.8, zorder=3)
    ax.axhline(ALWAYS_BLUE_WP, color='#b0b8c1', linestyle='--', linewidth=1.4,
               zorder=2, label=f'Always-Blue baseline ({ALWAYS_BLUE_WP:.3f})')
    ax.axhline(0.5, color='#e74c3c', linestyle=':', linewidth=1.2,
               zorder=2, label='Win/loss threshold (0.5)')
    _label_bars(ax, bars)
    ax.set_ylim(0.46, 0.62)
    ax.set_ylabel('Mean Blue Win Probability', fontsize=11)
    ax.set_title('Blue-Side Drafting Strength\n'
                 '(Trained Blue vs Random Red — 500 simulations)', fontsize=12)
    ax.legend(fontsize=9)
    _style(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")


# ------------------------------------------------------------------
# Figure 2 — Red-side drafting strength (expressed as Red WP)
# ------------------------------------------------------------------

def fig_red_strength(outpath):
    # Convert Blue WP to Red WP so higher = stronger Red drafting
    labels = ['Always-Blue\nBaseline', 'Winner MLP', 'Winner LSTM', 'Winner XGBoost']
    colors = [COLORS['random'], COLORS['mlp'], COLORS['lstm'], COLORS['xgb']]
    red_wps = [
        ALWAYS_RED_WP,
        1 - RESULTS['random_vs_mlp_red'][0],
        1 - RESULTS['random_vs_lstm_red'][0],
        1 - RESULTS['random_vs_xgb_red'][0],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, red_wps, color=colors,
                  edgecolor='white', linewidth=0.8, zorder=3)
    ax.axhline(ALWAYS_RED_WP, color='#b0b8c1', linestyle='--', linewidth=1.4,
               zorder=2, label=f'Always-Blue baseline ({ALWAYS_RED_WP:.3f})')
    ax.axhline(0.5, color='#e74c3c', linestyle=':', linewidth=1.2,
               zorder=2, label='Win/loss threshold (0.5)')
    _label_bars(ax, bars)
    ax.set_ylim(0.46, 0.62)
    ax.set_ylabel('Mean Red Win Probability', fontsize=11)
    ax.set_title('Red-Side Drafting Strength\n'
                 '(Random Blue vs Trained Red — 500 simulations)', fontsize=12)
    ax.legend(fontsize=9)
    _style(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")


# ------------------------------------------------------------------
# Figure 3 — Head-to-head balance
# ------------------------------------------------------------------

def fig_head_to_head(outpath):
    hth_vals   = [RESULTS['mlp_vs_mlp'][0],
                  RESULTS['lstm_vs_lstm'][0],
                  RESULTS['xgb_vs_xgb'][0]]
    hth_labels = ['Winner MLP\nvs MLP',
                  'Winner LSTM\nvs LSTM',
                  'Winner XGBoost\nvs XGBoost']
    hth_colors = [COLORS['mlp'], COLORS['lstm'], COLORS['xgb']]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(hth_labels, hth_vals, color=hth_colors,
                  edgecolor='white', linewidth=0.8, zorder=3)
    ax.axhline(0.5, color='#e74c3c', linestyle=':', linewidth=1.4,
               zorder=2, label='Balanced (0.5)')
    ax.axhline(BASELINE, color='#b0b8c1', linestyle='--', linewidth=1.2,
               zorder=2, label=f'Random baseline ({BASELINE:.3f})')
    _label_bars(ax, bars)
    ax.set_ylim(0.44, 0.57)
    ax.set_ylabel('Mean Blue Win Probability', fontsize=11)
    ax.set_title('Head-to-Head Results\n'
                 '(Same model on both sides — 500 simulations)', fontsize=12)
    ax.legend(fontsize=9)
    _style(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")


# ------------------------------------------------------------------
# Figure 4 — Cross-model heatmap
# ------------------------------------------------------------------

def fig_heatmap(outpath):
    # Rows = Blue model, Cols = Red model  [MLP, LSTM, XGBoost]
    # Only upper triangle is populated (lower matchups not evaluated)
    matrix = np.array([
        [RESULTS['mlp_vs_mlp'][0],
         RESULTS['mlp_blue_vs_lstm_red'][0],
         RESULTS['mlp_blue_vs_xgb_red'][0]],
        [np.nan,
         RESULTS['lstm_vs_lstm'][0],
         RESULTS['lstm_blue_vs_xgb_red'][0]],
        [np.nan,
         np.nan,
         RESULTS['xgb_vs_xgb'][0]],
    ])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.46, vmax=0.57, aspect='auto')
    labels = ['MLP', 'LSTM', 'XGBoost']
    ax.set_xticks([0, 1, 2]);  ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0, 1, 2]);  ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Red Model', fontsize=11)
    ax.set_ylabel('Blue Model', fontsize=11)
    ax.set_title('Cross-Model Mean Blue WP Heatmap\n'
                 '(Green = Blue advantage, Red = Red advantage)', fontsize=11)
    for i in range(3):
        for j in range(3):
            if not np.isnan(matrix[i, j]):
                color = 'white' if matrix[i, j] < 0.487 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.3f}',
                        ha='center', va='center',
                        fontsize=12, fontweight='bold', color=color)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                        fontsize=10, color='#aaa')
    plt.colorbar(im, ax=ax, label='Mean Blue Win Probability')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")


# ------------------------------------------------------------------
# Figure 5 — Percentage of games favoured
# ------------------------------------------------------------------

def fig_fav_pct(outpath):
    m_labels = ['Winner MLP', 'Winner LSTM', 'Winner XGBoost']
    x = np.arange(len(m_labels))
    width = 0.25

    blue_fav = [RESULTS['mlp_blue_vs_random'][1],
                RESULTS['lstm_blue_vs_random'][1],
                RESULTS['xgb_blue_vs_random'][1]]
    hth_fav  = [RESULTS['mlp_vs_mlp'][1],
                RESULTS['lstm_vs_lstm'][1],
                RESULTS['xgb_vs_xgb'][1]]
    # Red-favoured % = 100 - Blue-favoured % (random Blue vs trained Red)
    red_fav  = [100 - RESULTS['random_vs_mlp_red'][1],
                100 - RESULTS['random_vs_lstm_red'][1],
                100 - RESULTS['random_vs_xgb_red'][1]]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width, blue_fav, width,
                label='Blue model vs Random Red (Blue fav%)',
                color='#4e9af1', zorder=3)
    b2 = ax.bar(x,         hth_fav,  width,
                label='Head-to-head (Blue fav%)',
                color='#2ec4b6', zorder=3)
    b3 = ax.bar(x + width, red_fav,  width,
                label='Random Blue vs Red model (Red fav%)',
                color='#f4a261', zorder=3)

    ax.axhline(RESULTS['mlp_blue_vs_random'][1] * 0,   # dummy — use explicit lines
               color='none')
    ax.axhline(38.6, color='#b0b8c1', linestyle='--', linewidth=1.2,
               zorder=2, label='Random baseline (38.6%)')
    ax.axhline(50.0, color='#e74c3c', linestyle=':', linewidth=1.2,
               zorder=2, label='50% threshold')

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f'{bar.get_height():.0f}%',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(m_labels, fontsize=10)
    ax.set_ylim(30, 80)
    ax.set_ylabel('% Games Favoured', fontsize=11)
    ax.set_title('Percentage of Games Favoured by Side\n'
                 '(500 simulations per matchup)', fontsize=12)
    ax.legend(fontsize=8.5, loc='upper left')
    _style(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate adversarial evaluation figures")
    parser.add_argument('--outdir', default='.', help="Directory to save figures")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def path(name):
        return os.path.join(args.outdir, name)

    fig_blue_strength(path('adv_fig1_blue.png'))
    fig_red_strength( path('adv_fig2_red.png'))
    fig_head_to_head( path('adv_fig3_hth.png'))
    fig_heatmap(      path('adv_fig4_heatmap.png'))
    fig_fav_pct(      path('adv_fig5_favpct.png'))

    print(f"\nAll figures saved to '{args.outdir}/'")


if __name__ == '__main__':
    main()