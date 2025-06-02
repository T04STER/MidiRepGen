import os
import numpy as np
import pretty_midi
import torch


def parse_midi_file_to_tensor(
    midi_file_path: str, frame_per_second: int = 128
) -> torch.Tensor:
    """
    Parses a MIDI file and converts it to a tensor representation.

    Args:
        midi_file_path (str): Path to the MIDI file.

    Returns:
        torch.Tensor: A tensor representation of the MIDI file.
    """
    if not os.path.exists(midi_file_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_file_path}")
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        instrument = midi_data.instruments[0] if midi_data.instruments else None
        if not instrument:
            raise ValueError("No instruments found in the MIDI file.")

        piano_roll = instrument.get_piano_roll(fs=frame_per_second)

        return torch.tensor(piano_roll.T, dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"Error parsing MIDI file {midi_file_path}: {e}")


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
