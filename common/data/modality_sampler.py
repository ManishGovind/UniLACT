import random
from torch.utils.data import DataLoader, ConcatDataset , WeightedRandomSampler


class RandomModalityIterator:
    def __init__(self, dataloader_dict, total_batches, mode='round_robin'):
        """
        dataloader_dict: dict of modality_name -> dataloader
        total_batches: total number of batches to yield
        mode: 'round_robin' or 'random'
        """
        self.dataloaders = dataloader_dict
        self.iterators = {k: iter(dl) for k, dl in dataloader_dict.items()}
        self.modalities = list(dataloader_dict.keys())
        self.total_batches = total_batches
        self.counter = 0
        self.mode = mode
        self.round_index = 0

    def __next__(self):
        if self.counter >= self.total_batches:
            raise StopIteration

        # Decide which modality to use this time
        if self.mode == "round_robin":
            modality = self.modalities[self.round_index % len(self.modalities)]
            self.round_index += 1
        elif self.mode == "random":
            modality = random.choice(self.modalities)
        else:
            raise ValueError("Invalid mode. Use 'round_robin' or 'random'.")

        self.counter += 1

        try:
            batch = next(self.iterators[modality])
        except StopIteration:
            raise StopIteration  

        return batch




class RandomModalityBatchDataloader:
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.modalities = list(dataloader_dict.keys())
        self.dataset = ConcatDataset([dl.dataset for dl in dataloader_dict.values()])
        self.total_batches = min(len(dl) for dl in dataloader_dict.values()) * len(self.modalities)

    def __iter__(self):
        return RandomModalityIterator(self.dataloader_dict, self.total_batches)

    def __len__(self):
        return self.total_batches