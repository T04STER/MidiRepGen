import os
from torch.utils.data import Dataset

from src.dataloader.midi_parser import parse_midi_file_to_tensor


class MidiDataset(Dataset):
    def __init__(self, midi_files_dir, frame_per_second: int = 128):
        """
        Initializes the MidiDataset with a directory containing MIDI files.

        Args:
            midi_files_dir (str): Path to the directory containing MIDI files.
        """
        self.midi_files_dir = midi_files_dir
        self.midi_files = [f for f in os.listdir(midi_files_dir) if f.endswith(".mid")]
        self._fps = frame_per_second
        if not self.midi_files:
            raise ValueError(f"No MIDI files found in directory: {midi_files_dir}")

    def __len__(self):
        """
        Returns the number of MIDI files in the dataset.

        Returns:
            int: Number of MIDI files.
        """
        return len(self.midi_files)

    def __getitem__(self, idx):
        """
        Retrieves a MIDI file by index.

        Args:
            idx (int): Index of the MIDI file to retrieve.

        Returns:
            str: Path to the MIDI file.
        """
        if idx < 0 or idx >= len(self.midi_files):
            raise IndexError("Index out of range.")

        midi_file_path = os.path.join(self.midi_files_dir, self.midi_files[idx])
        return parse_midi_file_to_tensor(midi_file_path, frame_per_second=self._fps)
