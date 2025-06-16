from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.dataloader.dataset import MidiDataset

class MidiDataLoader(DataLoader):
    def __init__(self, dataset: MidiDataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False, padding_value=0.0):
        """
        Initializes the MIDI DataLoader.

        Args:
            dataset: The dataset to load.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at every epoch.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        """
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=self._collate_fn_impl)
        self.padding_value = padding_value
    
    def _collate_fn_impl(self, batch):
        """
            Pads the batch of MIDI tensors to ensure they have the same length.
        """
        padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
        return padded
