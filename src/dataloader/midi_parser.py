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

    :param midi_file_path: Path to the MIDI file.
    :type midi_file_path: str
    :param frame_per_second: Sampling frequency for piano roll.
    :type frame_per_second: int
    :param strip_bounds: Whether to strip silent frames from the start and end.
    :type strip_bounds: bool
    :param strip_pitch: Range of pitches to retain (start, end).
    :type strip_pitch: tuple[int, int], optional
    :return: Sparse representation of the MIDI piano roll (time x pitch).
    :rtype: scipy.sparse.csr_matrix
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

    :param arr: 2D array (time x pitch)
    :type arr: np.ndarray
    :return: Stripped array or None if fully silent.
    :rtype: np.ndarray or None
    """
    if arr is None or arr.ndim != 2:
        return None

    non_zero_rows = np.any(arr != 0, axis=1)
    if not np.any(non_zero_rows):
        return None

    start_idx = np.argmax(non_zero_rows)
    end_idx = len(non_zero_rows) - np.argmax(non_zero_rows[::-1])
    return arr[start_idx:end_idx]


def piano_roll_to_pretty_midi(piano_roll, fs=64, path="eg.mid", pitch_range=(24, 84)):
    """
    Convert a piano roll (tensor or array) to a MIDI file using pretty_midi.
    Expects piano_roll to be shape [time, pitch] and in range [0, 1].
    """
    if isinstance(piano_roll, torch.Tensor):
        piano_roll = (piano_roll * 127).clamp(0, 127).detach().cpu().numpy().astype(np.int32)
    
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    
    pitch_offset = pitch_range[0]
    time_steps, pitch_count = piano_roll.shape
    
    for pitch_idx in range(pitch_count):
        pitch = pitch_idx + pitch_offset
        
        note_on = False
        note_start = 0
        
        for time in range(time_steps):
            velocity = piano_roll[time, pitch_idx]
            
            if velocity > 0 and not note_on:
                note_on = True
                note_start = time
                note_velocity = int(velocity)
            elif velocity == 0 and note_on:
                note_on = False
                note = pretty_midi.Note(
                    velocity=note_velocity,
                    pitch=pitch,
                    start=note_start / fs,
                    end=time / fs
                )
                instrument.notes.append(note)
        
        if note_on:
            note = pretty_midi.Note(
                velocity=note_velocity,
                pitch=pitch,
                start=note_start / fs,
                end=time_steps / fs
            )
            instrument.notes.append(note)
    
    pm.instruments.append(instrument)
    pm.write(path)



def parse_midi_file_to_pitch_start_end_tensor(midi_file_path: str) -> Optional[np.ndarray]:
    """
    Parses a MIDI file and converts it to a tensor representation of pitch, start, and end times.

    :param midi_file_path: Path to the MIDI file.
    :type midi_file_path: str

    :return: A tensor representation of the MIDI file with shape [N, 4],
             where N is the number of notes, and each row contains [pitch, velocity, delta_start, duration].
             Returns None if no notes are found.
    :rtype: np.ndarray or None
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append([note.pitch, note.velocity, note.start, note.end ])

    if not notes:
        return None

    # encode start and end time to as delta of previous note and duration
    notes.sort(key=lambda x: (x[2], x[0]))
    delta_time_notes = []
    for i in range(len(notes)):
        if i == 0:
            note = notes[i]
            duration = note[3] - note[2]
            delta_time_notes.append([note[0], note[1], note[2], duration])
        else:
            note = notes[i]
            prev_note = notes[i - 1]
            # delta betwen previous start and current start
            delta_time = note[2] - prev_note[2]
            duration = note[3] - note[2]
            delta_time_notes.append([note[0], note[1], delta_time, duration])
            
    notes_np = np.array(delta_time_notes, dtype=np.float32)
    # normalize velocity to [0, 1], pitch stays as ordinal value will use embedding
    notes_np[:, 1] = notes_np[:, 1] / 127.0
    return notes_np
    

