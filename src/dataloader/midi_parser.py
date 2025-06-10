import os
from typing import Optional, Tuple
import numpy as np
import pretty_midi
import torch
from scipy import sparse



def parse_midi_file_to_sparse(
    midi_file_path: str,
    frame_per_second: int = 128,
    strip_bounds: bool = True,
    strip_pitch: Optional[Tuple[int, int]] = None
) -> Optional[sparse.csr_matrix]:
    """
    Parses a MIDI file and converts it to a sparse matrix representation.

    Args:
        midi_file_path (str): Path to the MIDI file.
        frame_per_second (int): Sampling frequency for piano roll.
        strip_bounds (bool): Whether to strip silent frames from the start and end.
        strip_pitch (Tuple[int, int], optional): Range of pitches to retain (start, end).

    Returns:
        scipy.sparse.csr_matrix: Sparse representation of the MIDI piano roll (time x pitch).
    """
    if not os.path.exists(midi_file_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_file_path}")
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        instrument = midi_data.instruments[0] if midi_data.instruments else None
        if not instrument:
            raise ValueError("No instruments found in the MIDI file.")

        piano_roll = instrument.get_piano_roll(fs=frame_per_second).T/127  # shape: (time, pitch) + normalization to [0, 1]
        

        if strip_bounds:
            piano_roll = _strip_numpy_array(piano_roll)

        if strip_pitch is not None and piano_roll is not None:
            start_freq, end_freq = strip_pitch
            piano_roll = piano_roll[:, start_freq:end_freq]

        if piano_roll is None or piano_roll.size == 0:
            return None

        return sparse.csr_matrix(piano_roll)

    except Exception as e:
        print(f"Error parsing MIDI file {midi_file_path}: {e}")
        return None


def _strip_numpy_array(arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Removes leading and trailing rows with only zeros (i.e., silence).

    Args:
        arr (np.ndarray): 2D array (time x pitch)

    Returns:
        np.ndarray: Stripped array or None if fully silent.
    """
    if arr is None or arr.ndim != 2:
        return None

    non_zero_rows = np.any(arr != 0, axis=1)
    if not np.any(non_zero_rows):
        return None

    start_idx = np.argmax(non_zero_rows)
    end_idx = len(non_zero_rows) - np.argmax(non_zero_rows[::-1])
    return arr[start_idx:end_idx]

def parse_tensor_to_midi_file(
    tensor: torch.Tensor, output_file_path: str, frame_per_second: int = 128
):
    """
    Converts a tensor representation of a piano roll back to a MIDI file.

    Args:
        tensor (torch.Tensor): A [T, 128] piano roll tensor.
        output_file_path (str): Path to save the output MIDI file.
        frame_per_second (int): Sampling rate for the piano roll.
    """
    try:
        piano_roll = tensor.cpu().numpy().T
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        for pitch in range(piano_roll.shape[0]):
            velocity_track = piano_roll[pitch]
            is_active = velocity_track > 0

            padded = np.pad(is_active.astype(np.int8), (1, 1))
            changes = np.diff(padded)

            note_on_times = np.where(changes == 1)[0]
            note_off_times = np.where(changes == -1)[0]

            for start, end in zip(note_on_times, note_off_times):
                velocity = int(np.max(velocity_track[start:end]) * 127)
                note = pretty_midi.Note(
                    velocity=velocity if velocity > 0 else 100,
                    pitch=pitch,
                    start=start / frame_per_second,
                    end=end / frame_per_second,
                )
                instrument.notes.append(note)

        midi.instruments.append(instrument)
        midi.write(output_file_path)
    except Exception as e:
        raise ValueError(f"Error writing MIDI file {output_file_path}: {e}")



def parse_midi_file_to_pitch_start_end_tensor(midi_file_path: str) -> torch.Tensor:
    """
    Parses a MIDI file and converts it to a tensor representation of pitch, start, and end times.

    Args:
        midi_file_path (str): Path to the MIDI file.

    Returns:
        torch.Tensor: A tensor representation of the MIDI file with shape [N, 3],
                      where N is the number of notes, and each row contains [pitch, start, end].
    """
    def parse_midi_file(midi_file_path):
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append([note.pitch, note.start, note.end, note.velocity])
        return np.array(notes)

    def convert_to_tensor(notes):
        return torch.tensor(notes, dtype=torch.float32)

    return convert_to_tensor(parse_midi_file(midi_file_path))
