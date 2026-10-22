"""Auditable held-out errors and a spatial comparison against training-source labels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_holdout(points, train, test, predictions, shape, out_dir):
    frame = pd.DataFrame({'row': points.rows, 'col': points.cols, 'block': points.block,
                          'reference_label': points.y, 'split': 'train'})
    frame.loc[test, 'split'] = 'test'
    frame['prediction'] = np.nan
    frame.loc[test, 'prediction'] = predictions
    frame.to_csv(out_dir / 'sample_assignments.csv', index=False)
    held = frame.iloc[test].copy()
    held['error'] = np.where(held.reference_label == predictions, 'agreement',
                            np.where(predictions == 1, 'false_water', 'missed_water'))
    held.to_csv(out_dir / 'heldout_predictions.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    axes[0].scatter(points.cols[train], points.rows[train], s=3, c='#aaaaaa', label='training')
    axes[0].scatter(points.cols[test], points.rows[test], s=4, c='#8a51a5', label='held out')
    axes[0].legend(loc='upper right', markerscale=3)
    for ax, values, title in [(axes[1], points.y[test], 'Held-out JRC-derived labels'),
                              (axes[2], predictions, 'Decision Tree predictions')]:
        for code, colour, name in [(0, '#a18a6b', 'land'), (1, '#005eb8', 'water')]:
            mask = values == code
            ax.scatter(points.cols[test][mask], points.rows[test][mask], s=5, c=colour, label=name)
        ax.set_title(title)
        ax.legend(loc='upper right', markerscale=3)
    axes[0].set_title('Spatial train/test separation')
    wrong = predictions != points.y[test]
    axes[2].scatter(points.cols[test][wrong], points.rows[test][wrong], s=22,
                    facecolors='none', edgecolors='#d73027', linewidths=.8)
    for ax in axes:
        ax.set_xlim(0, shape[1]); ax.set_ylim(shape[0], 0)
        ax.set_xlabel('Raster column'); ax.set_aspect('equal')
    axes[0].set_ylabel('Raster row')
    fig.suptitle('Agreement with historical labels; red rings mark disagreements, not verified ground errors')
    fig.tight_layout()
    fig.savefig(out_dir / 'spatial_validation.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
