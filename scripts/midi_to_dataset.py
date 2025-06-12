"""
    Selects random sample of MIDI files to tensors. 
    supports two modes:
    - piano roll 
        Embeds the MIDI file as a piano roll tensor. With 64 Frames per second,
    - note events:
        Each note event is represented as [pitch, velocity, start_time, end_time].
    - note events:
        Each note event is represented as [pitch, velocity, delta_time, silence].
"""
import argparse
import os
from pathlib import Path
import pickle
import random
import torch
from src.dataloader.dataset import PianoRollMidiDataset, EventMidiDataset
from src.common.config_utils import get_config

def subsample(records, sample_rate=0.1):
    """
        Selects a random sample of MIDI files from the dataset and converts them to tensors.
        :param records: List of MIDI file paths.
        :param sample_rate: Fraction of records to sample (between 0 and 1).
        :return: List of MIDI file paths sampled from the dataset.
    """
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("Sample rate must be between 0 and 1.")
    
    sample_size = int(len(records) * sample_rate)
    return random.sample(records, sample_size)



def get_argparser():
    parser = argparse.ArgumentParser(description="Convert MIDI files to tensor representations.")
    parser.add_argument("--mode", type=str, choices=["piano_roll", "note_events"], default="piano_roll",)
    return parser


def piano_roll_dataset(sampled_midi_files_dir, out_path:str, frame_per_second=64):
    pitch_to_strip = (24, 84)  # Take  only pitches between 24 and 84 C1 to C6 (https://arxiv.org/pdf/1809.07600)

    dataset = PianoRollMidiDataset(
        midi_files=sampled_midi_files_dir,
        frame_per_second=frame_per_second,
        verbose=True,
        strip_bounds=True,
        pitch_to_strip=pitch_to_strip
    )

    pickle.dump(
        dataset,
        open(out_path, "wb"),
    )
    print(f"Dataset saved to {out_path}")
    

def event_based_dataset(sampled_midi_files_dir, out_path:str, note_count=None):
    if note_count is None:
        note_count = 128
    dataset = EventMidiDataset(sampled_midi_files_dir, verbose=True, note_count=note_count)

    pickle.dump(
        dataset,
        open(out_path, "wb"),
    )


def main(args):
    config = get_config()
    if seed := config.get("random_seed"):
        random.seed(seed)
        torch.manual_seed(seed)
    
    dataset_config = config["dataset"]
    midi_files_dir_str = dataset_config["out_unpacked"]
    files_dir = Path(midi_files_dir_str) / "midis"
    if not files_dir.exists():
        raise FileNotFoundError(f"MIDI files directory does not exist: {midi_files_dir_str}")
    midi_all_files = [f for f in files_dir.glob("*.mid") if f.is_file()]
    if not midi_all_files:
        raise ValueError(f"No MIDI files found in directory: {midi_files_dir_str}")
    
    preprocess_config = config.get("preprocess", {})
    sample_rate = preprocess_config.get("sample_rate", 0.1)
    midi_files = subsample(midi_all_files, sample_rate=sample_rate)
    midi_files = [str(f) for f in midi_files]
    match args.mode:
        case "piano_roll":
            out_path = preprocess_config.get("out_piano_roll")
            if not out_path:
                raise ValueError("Output path for piano roll dataset is not specified in the config.")
            if not out_path.endswith(".pkl"):
                raise ValueError("Output path for piano roll dataset must end with .pkl")
            piano_roll_dataset(midi_files, out_path, frame_per_second=preprocess_config.get("frame_per_second", 64))
        case "note_events":
            out_path = preprocess_config.get("out_note_events")
            if not out_path:
                raise ValueError("Output path for note events dataset is not specified in the config.")
            if not out_path.endswith(".pkl"):
                raise ValueError("Output path for note events dataset must end with .pkl")
            event_based_dataset(midi_files, out_path, note_count=preprocess_config.get("note_count"))
        case _:
            raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":    
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
    