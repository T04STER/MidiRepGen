from functools import lru_cache
import os
from typing import Optional
from scipy import sparse
import torch
from torch.utils.data import Dataset
import tqdm

from src.dataloader.midi_parser import parse_midi_file_to_sparse, parse_midi_file_to_pitch_start_end_tensor


class PianoRollMidiDataset(Dataset):
    def __init__(self, midi_files: list[str], frame_per_second: int = 64, window_len=10, verbose: bool = False, strip_bounds: bool = True, pitch_to_strip: Optional[tuple] = None):
        """
        Initializes the MidiDataset with a directory containing MIDI files.
        Splits it into time windows with specified duration and frame rate.

        :param midi_files: List of paths to MIDI files.
        :type midi_files: list[str]

        :param frame_per_second: Frame rate for the MIDI files.
        :type frame_per_second: int

        :param window_len: Length of the time window in seconds.
        :type window_len: int

        :param verbose: If True, prints information about the dataset.
        :type verbose: bool

        :param strip_bounds: If True, strips zero frames from the beginning and end of the MIDI files.
        :type strip_bounds: bool

        :param strip_frequencies: Specifies pitch and velocities to ignore.
        :type strip_frequencies: tuple
        """
        self._fps = frame_per_second
        if not midi_files:
            raise ValueError("The list of MIDI files is empty.")
            
        self.verbose = verbose
        if verbose:
            print(f"Got {len(midi_files)} MIDI files in, frame rate set to {self._fps} FPS.")

        self.strip_bounds = strip_bounds
        self.strip_frequencies = pitch_to_strip if pitch_to_strip else (None, None)
        self.window_len = window_len
        self.midi_sparse_list: list[sparse.csr_matrix] = []  # for efficient storage of MIDI tensors
        self._init_midi_tensors(midi_files)


    def _split_into_time_windows(self, sparse_matrix: sparse.csr_matrix) -> list[sparse.csr_matrix]:
        """
        Splits the sparse matrix into time windows of specified length.

        :param sparse_matrix: The preprocessed sparse matrix to split (shape: [time, pitch]).
        :type sparse_matrix: scipy.sparse.csr_matrix

        :return: List of sparse matrices, each representing a time window.
        :rtype: list[scipy.sparse.csr_matrix]
        """
        window_size = int(self.window_len * self._fps)
        total_rows = sparse_matrix.shape[0]

        return [
            sparse_matrix[i:i + window_size]
            for i in range(0, total_rows - window_size + 1, window_size)
        ]



    def _init_midi_tensors(self, midi_files: list[str]):
        """
        Initializes the MIDI tensors by parsing each MIDI file and converting it to a tensor representation.
        """
        iter_ = tqdm.tqdm(midi_files) if self.verbose else midi_files
        for midi_file in iter_:
            midi_sparse = parse_midi_file_to_sparse(
                midi_file,
                frame_per_second=self._fps,
                strip_bounds=self.strip_bounds,
                strip_pitch=self.strip_frequencies
            )
            if midi_sparse is None or midi_sparse.shape[0] == 0:
                if self.verbose:
                    print(f"Warning: MIDI file {midi_file} could not be parsed or is empty.")
            else:
                splited_midi = self._split_into_time_windows(midi_sparse)
                self.midi_sparse_list.extend(splited_midi)
        
        if self.verbose:
            print(f"Initialized {len(self.midi_sparse_list)} MIDI tensors.")

    def __len__(self):
        return len(self.midi_sparse_list)

    @lru_cache(maxsize=1024)
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.midi_sparse_list):
            raise IndexError("Index out of range.")
        midi_sparse =  self.midi_sparse_list[idx]
        tensor = torch.tensor(midi_sparse.toarray(), dtype=torch.float32)
        return tensor




class MidiDatasetV2(Dataset):
    def __init__(self, midi_files_dir, verbose: bool = False, strip_bounds: bool = True):
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
        if not self.midi_files:
            raise ValueError(f"No MIDI files found in directory: {midi_files_dir}")
        self.verbose = verbose
        if verbose:
            print(f"Found {len(self.midi_files)} MIDI files in {midi_files_dir}")

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
        midi_tensor = parse_midi_file_to_pitch_start_end_tensor(midi_file_path)
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
        debug_clip_range = min(100, end_idx - start_idx)
        end_idx = start_idx + debug_clip_range
        return tensor[start_idx:end_idx]
        
        