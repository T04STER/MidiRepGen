from functools import lru_cache
import os
from typing import Optional
import numpy as np
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




class EventMidiDataset(Dataset):
    def __init__(self, midi_files: list[str], note_count=10, verbose: bool = False):
        """
        Initializes the EveMidiDataset with a directory containing MIDI files.
        Splits it into time windows with specified note count.

        :param midi_files: List of paths to MIDI files.
        :type midi_files: list[str]

        :param note_count: Count of notes to split into a widnow.
        :type note_count: int

        :param verbose: If True, prints information about the dataset.
        :type verbose: bool

        """
        if not midi_files:
            raise ValueError("The list of MIDI files is empty.")

        self._note_count = note_count            
        self.verbose = verbose
        if verbose:
            print(f"Got {len(midi_files)} MIDI files")

        self.midi_list: list[np.array] = []  # more efficient storage of MIDI tensors
        self._init_midi_tensors(midi_files)


    def _split_into_time_windows(self, event_tensor: np.ndarray) -> list[np.ndarray]:
        """
        Splits the tensor into time windows of specified length.

        :param event_tensor: The preprocessed tensor to split (shape: [time, pitch]).
        :return: List of tensors, each representing a time window.
        """
        total_rows = event_tensor.shape[0]
        return [
            event_tensor[i:i + self._note_count]
            for i in range(0, total_rows - self._note_count, self._note_count)
        ]



    def _init_midi_tensors(self, midi_files: list[str]):
        """
        Initializes the MIDI tensors by parsing each MIDI file and converting it to a tensor representation.
        """
        iter_ = tqdm.tqdm(midi_files) if self.verbose else midi_files
        for midi_file in iter_:
            midi_tensor = parse_midi_file_to_pitch_start_end_tensor(midi_file)
            if midi_tensor is None or midi_tensor.shape[0] == 0:
                if self.verbose:
                    print(f"Warning: MIDI file {midi_file} could not be parsed or is empty.")
            else:
                splited_midi = self._split_into_time_windows(midi_tensor)
                self.midi_list.extend(splited_midi)
        
        if self.verbose:
            print(f"Initialized {len(self.midi_list)} MIDI tensors.")

    def __len__(self):
        return len(self.midi_list)

    @lru_cache(maxsize=1024)
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.midi_list):
            raise IndexError("Index out of range.")
        np_array = self.midi_list[idx]
        return torch.tensor(np_array, dtype=torch.float32)
