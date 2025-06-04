import os
import torch
from torch.utils.data import Dataset

from src.dataloader.midi_parser import parse_midi_file_to_tensor


class MidiDataset(Dataset):
    def __init__(self, midi_files_dir, frame_per_second: int = 128, verbose: bool = False, strip_bounds: bool = True):
        """
        Initializes the MidiDataset with a directory containing MIDI files.

        Args:
            midi_files_dir (str): Path to the directory containing MIDI files.
            frame_per_second (int): Frame rate for the MIDI files.
            verbose (bool): If True, prints information about the dataset.
            strip_bounds (bool): If True, strips zeroes frames from the beginning and end of the MIDI files.
        """
        self.midi_files_dir = midi_files_dir
        self.midi_files = [f for f in os.listdir(midi_files_dir) if f.endswith(".mid")]
        self._fps = frame_per_second
        if not self.midi_files:
            raise ValueError(f"No MIDI files found in directory: {midi_files_dir}")
        self.verbose = verbose
        if verbose:
            print(f"Found {len(self.midi_files)} MIDI files in {midi_files_dir}, frame rate set to {self._fps} FPS.")

        self.strip_bounds = strip_bounds

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
        midi_tensor = parse_midi_file_to_tensor(midi_file_path, frame_per_second=self._fps)
        if self.strip_bounds:
            midi_tensor = self.strip_tensor(midi_tensor)
        return midi_tensor

    def strip_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Strips zero frames from the beginning and end of the tensor.

        Args:
            tensor (torch.Tensor): The tensor to strip.

        Returns:
            torch.Tensor: The stripped tensor.
        """
        non_zero_mask = tensor.abs().sum(dim=tuple(range(1, tensor.dim()))) != 0
        non_zero_indices = non_zero_mask.nonzero(as_tuple=False).squeeze()
        if non_zero_indices.numel() == 0:
            if self.verbose:
                print("Warning: Tensor is empty after stripping.")
            return tensor
        
        start_idx = non_zero_indices[0].item()
        end_idx = non_zero_indices[-1].item() + 1
        print(f"Stripping tensor from {start_idx} to {end_idx}") if self.verbose else None
        return tensor[start_idx:end_idx]
        
        