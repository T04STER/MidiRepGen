import numpy as np
import torch


def iou_intervals(pred_start, pred_duration, target_start, target_duration):
    pred_end = pred_start + pred_duration
    target_end = target_start + target_duration
    
    inter_start = np.maximum(pred_start, target_start)
    inter_end = np.minimum(pred_end, target_end)
    intersection = np.maximum(0, inter_end - inter_start)

    union = np.maximum(pred_end, target_end) - np.minimum(pred_start, target_start)

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0, intersection / union, 0.0)

    return iou


def unoffset_notes_start(notes: np.ndarray | torch.Tensor) -> np.ndarray:
    """Unoffsets start time"""
    if isinstance(notes, torch.Tensor):
        notes = notes.detach().cpu().numpy()
    if notes.shape[1] < 4:
        raise ValueError("Notes must have at least 4 columns (pitch, start, duration, velocity)")
    
    unoffset_start = np.cumsum(notes[:, 2])
    unoffset_notes = np.copy(notes)
    unoffset_notes[:, 2] = unoffset_start
    return unoffset_notes

def pitch_match_accuracy_iou_ordered(pred_notes: np.ndarray | torch.Tensor, target_notes: np.ndarray|torch.Tensor, iou_threshold=0.1):
    """
        Computes the pitch match accuracy and IoU for ordered notes. 
        Args:
            pred_notes (np.ndarray or torch.Tensor): Predicted notes of shape (N, 4) where each row is [pitch, velocity, start, duration].
            target_notes (np.ndarray or torch.Tensor): Target notes of shape (M, 4) with the same format.
            iou_threshold (float): IoU threshold for matching.
    """
    if isinstance(pred_notes, torch.Tensor):
        pred_notes = pred_notes.detach().cpu().numpy()
    if isinstance(target_notes, torch.Tensor):
        target_notes = target_notes.detach().cpu().numpy()
    N_pred = pred_notes.shape[0]
    N_target = target_notes.shape[0]
    pred_notes = unoffset_notes_start(pred_notes)
    target_notes = unoffset_notes_start(target_notes)
    # Compare up to the min length (to avoid indexing errors)
    N = min(N_pred, N_target)
    
    pred_pitch = pred_notes[:N, 0]
    pred_start = pred_notes[:N, 2]
    pred_duration = pred_notes[:N, 3]
    
    target_pitch = target_notes[:N, 0]
    target_start = target_notes[:N, 2]
    target_duration = target_notes[:N, 3]

    pitch_matches = (pred_pitch == target_pitch)
    ious = iou_intervals(pred_start, pred_duration, target_start, target_duration)

    # Match if pitch matches and IoU above threshold
    # TODO: change matches to mAP (logits are required so another function maybee )
    matches = pitch_matches & (ious >= iou_threshold)
    
    accuracy = pitch_matches.sum() / N if N > 0 else 0.0
    return accuracy, ious, matches