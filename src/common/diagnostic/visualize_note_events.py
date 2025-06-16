from matplotlib import cm, pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
import numpy as np
import torch

from src.common.metrics.metrics import pitch_match_accuracy_iou_ordered


def plot_piano_roll_from_note_events(note_array: torch.Tensor|np.ndarray, figsize=(10, 4), title=None, fig=None, ax=None, show=True, show_colorbar=True, cmap='viridis'):
    if not isinstance(note_array, np.ndarray):
        note_array = note_array.detach().cpu().numpy()

    scaled_pitches = np.clip(note_array[:, 0], 0, 127).astype(np.int8)
    velocities = np.clip(note_array[:, 1] * 127, 0, 127).astype(np.int8)
    note_delta = note_array[:, 2]
    note_start = np.cumsum(note_delta)
    note_duration = note_array[:, 3]
    note_end = note_start + note_duration

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    norm = mcolors.Normalize(vmin=0, vmax=127)
    if cmap != 'viridis':
        cmap = cm.get_cmap(cmap)
    else:
        cmap = cm.viridis
    vel_color = cmap(norm(velocities))

    for i in range(len(note_start)):
        ax.hlines(scaled_pitches[i], note_start[i], note_end[i],
                  colors=vel_color[i], linewidth=10, alpha=0.5)
    if show_colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(velocities)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Velocity')
    
    ax.set_xlim(0, np.max(note_end) + 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch')
    ax.set_title('Piano Roll' if title is None else title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def compare_reco_true(true: torch.Tensor|np.ndarray, reco: torch.Tensor|np.ndarray, figsize=(10, 4), threshold=0.5):
    """
        Compares true and reconstructed piano rolls.
    """
    fig, ax = plt.subplots(figsize=figsize, sharex=True)
    plot_piano_roll_from_note_events(true, figsize=figsize, fig=fig, ax=ax, cmap='Blues', show=False)
    plot_piano_roll_from_note_events(reco, figsize=figsize, fig=fig, ax=ax, cmap='Reds', show=False)

    acc, iou, _ = pitch_match_accuracy_iou_ordered(
        reco, true, iou_threshold=threshold
    )
    ax.set_title('Comparison of True and Reconstructed Piano Rolls (Accuracy: {:.2f}, $\\overline{{IoU}} = {:.2f}$)'.format(acc, iou.mean()))
    legend_elements = [
        Patch(facecolor='blue', label='True'),
        Patch(facecolor='red', label='Reconstructed'),
    ]
    plt.legend(handles=legend_elements)
    plt.show()
