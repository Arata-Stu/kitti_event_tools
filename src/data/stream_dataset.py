from torch.utils.data import IterableDataset
from src.data.sequence_stream import SequenceForStreaming



class StreamingConcatDataset(IterableDataset):
    def __init__(self, sequence: SequenceForStreaming):
        self.sequence = sequence

    def __iter__(self):
        return iter(self.sequence)
    