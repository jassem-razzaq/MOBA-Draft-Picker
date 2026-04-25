"""
Generate all figures for the win probability model results section.

Produces three PNG files:
  fig1_accuracy_auc.png  — Accuracy and AUC-ROC side by side
  fig2_log_loss.png      — Log Loss bar chart
  fig3_normalised.png    — Normalised metric comparison

Requirements: matplotlib, numpy
  pip install matplotlib numpy

Usage:
  python generate_win_predictor_figures.py
  python generate_win_predictor_figures.py --outdir results/figures

To update with your own results, edit the RESULTS dictionary at the top.
"""
import argparse
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Data — update these values if you retrain the models
# ------------------------------------------------------------------

RESULTS = {
    'always_blue_baseline': {
        'accuracy': 0.5273,   # = Blue win rate in test set
        'auc':      0.5000,   # constant predictor has no discrimination
        'log_loss': 0.6834,   # entropy of the Blue win rate
    },
    'xgboost': {
        'accuracy': 0.5556,
        'auc':      0.5776,
        'log_loss': 0.6855,
    },
    'neural_network': {
        'accuracy': 0.5628,
        'auc':      0.5861,
        'log_loss': 0.7431,
    },
    'logistic': {
        'accuracy': 0.5758,
        'auc':      0.5965,
        'log_loss': 0.6912,
    },
}

# Display labels and colours (order = left to right in charts)
MODEL_ORDER  = ['always_blue_baseline', 'xgboost', 'neural_network', 'logistic']
MODEL_LABELS = ['Always-Blue\nBaseline', 'XGBoost', 'Neural\nNetwork', 'Logistic\nRegression']
BAR_COLORS   = ['#b0b8c1', '#4e9af1', '#f4a261', '#2ec4b6']


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def _label_bars(ax, bars, fmt='{:.3f}', offset=0.002):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                fmt.format(bar.get_height()),
                ha='center', va='bottom', fontsize=9, fontweight='bold')


# ------------------------------------------------------------------
# Figure 1 — Accuracy and AUC-ROC side by side
# ------------------------------------------------------------------

def fig_accuracy_auc(outpath):
    accuracy = [RESULTS[m]['accuracy'] for m in MODEL_ORDER]
    auc      = [RESULTS[m]['auc']      for m in MODEL_ORDER]
    baseline_acc = RESULTS['always_blue_baseline']['accuracy']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Win Probability Model Performance',
                 fontsize=14, fontweight='bold', y=1.02)

    # Left: Accuracy
    ax = axes[0]
    bars = ax.bar(MODEL_LABELS, accuracy, color=BAR_COLORS,
                  edgecolor='white', linewidth=0.8, zorder=3)
    ax.axhline(baseline_acc, color='#b0b8c1', linestyle='--', linewidth=1.4,
               zorder=2, label=f'Always-Blue baseline ({baseline_acc:.3f})')
    _label_bars(ax, bars)
    ax.set_ylim(0.48, 0.62)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Classification Accuracy', fontsize=12)
    ax.legend(fontsize=9)
    _style(ax)

    # Right: AUC-ROC
    ax = axes[1]
    bars = ax.bar(MODEL_LABELS, auc, color=BAR_COLORS,
                  edgecolor='white', linewidth=0.8, zorder=3)
    ax.axhline(0.5, color='#b0b8c1', linestyle='--', linewidth=1.4,
               zorder=2, label='Random classifier (AUC = 0.5)')
    _label_bars(ax, bars)
    ax.set_ylim(0.46, 0.63)
    ax.set_ylabel('AUC-ROC', fontsize=11)
    ax.set_title('AUC-ROC Score', fontsize=12)
    ax.legend(fontsize=9)
    _style(ax)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")


# ------------------------------------------------------------------
# Figure 2 — Log Loss
# ------------------------------------------------------------------

def fig_log_loss(outpath):
    log_losses = [RESULTS[m]['log_loss'] for m in MODEL_ORDER]

    # Naive log-loss: a predictor that always outputs the training Blue win rate
    blue_rate = RESULTS['always_blue_baseline']['accuracy']
    naive_ll  = -(blue_rate * math.log(blue_rate) +
                  (1 - blue_rate) * math.log(1 - blue_rate))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(MODEL_LABELS, log_losses, color=BAR_COLORS,
                  edgecolor='white', linewidth=0.8, zorder=3)
    ax.axhline(naive_ll, color='#b0b8c1', linestyle='--', linewidth=1.4,
               zorder=2,
               label=f'Naive baseline log loss ({naive_ll:.3f})')
    _label_bars(ax, bars)
    ax.set_ylim(0.64, 0.78)
    ax.set_ylabel('Log Loss  (lower = better)', fontsize=11)
    ax.set_title('Log Loss — Win Probability Model Performance',
                 fontsize=12)
    ax.legend(fontsize=9)
    _style(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")


# ------------------------------------------------------------------
# Figure 3 — Normalised metric comparison (excludes baseline)
# ------------------------------------------------------------------

def fig_normalised(outpath):
    """
    Normalise each metric to [0, 1] across the three trained models
    so all three can be compared on the same axis.

    Accuracy  : higher is better — normalise directly
    AUC-ROC   : higher is better — normalise directly
    Log Loss  : lower is better — invert before normalising
                (worst log loss maps to 0, best maps to 1)
    """
    trained = ['xgboost', 'neural_network', 'logistic']
    labels  = ['XGBoost', 'Neural Network', 'Logistic Regression']
    colors  = ['#4e9af1', '#f4a261', '#2ec4b6']

    acc_vals = [RESULTS[m]['accuracy'] for m in trained]
    auc_vals = [RESULTS[m]['auc']      for m in trained]
    ll_vals  = [RESULTS[m]['log_loss'] for m in trained]

    def normalise(vals, invert=False):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [0.5] * len(vals)
        normed = [(v - lo) / (hi - lo) for v in vals]
        return [1 - v for v in normed] if invert else normed

    acc_norm = normalise(acc_vals)
    auc_norm = normalise(auc_vals)
    ll_norm  = normalise(ll_vals, invert=True)   # lower log loss = better

    x = np.arange(3)   # three metric groups
    width = 0.22
    metric_labels = ['Accuracy', 'AUC-ROC', 'Log Loss\n(inverted)']

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, color) in enumerate(zip(labels, colors)):
        vals = [acc_norm[i], auc_norm[i], ll_norm[i]]
        ax.bar(x + i * width, vals, width,
               label=label, color=color, edgecolor='white', zorder=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Normalised score  (higher = better)', fontsize=10)
    ax.set_title('Normalised Metric Comparison Across Models\n'
                 '(0 = worst observed,  1 = best observed)', fontsize=11)
    ax.legend(fontsize=10)
    _style(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outpath}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate win probability model figures")
    parser.add_argument('--outdir', default='.',
                        help="Directory to save figures (default: current directory)")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def path(name):
        return os.path.join(args.outdir, name)

    fig_accuracy_auc(path('fig1_accuracy_auc.png'))
    fig_log_loss(    path('fig2_log_loss.png'))
    fig_normalised(  path('fig3_normalised.png'))

    print(f"\nAll figures saved to '{args.outdir}/'")


if __name__ == '__main__':
    main()