import torch_geometric.transforms as T
from torch import default_generator
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader



def get_dataloader(dataset, batch_size, data_split_ratio, seed=3407):
    """

    Args:
        dataset: which dataset you want
        batch_size: int
        data_split_ratio: list [train, valid, test]
        seed: random seed to split the dataset randomly

    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    num_train = int(data_split_ratio[0] * len(dataset))
    num_eval = int(data_split_ratio[1] * len(dataset))
    num_test = len(dataset) - num_train - num_eval

    train, eval, test = random_split(dataset,
                                     lengths=[num_train, num_eval, num_test],
                                     generator=default_generator.manual_seed(seed))

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=True)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=True)

    return dataloader


def get_dataset(dataset_root, dataset_name):
    if dataset_name.lower() in ['mutagenicity', 'enzymes']:
        return TUDataset(dataset_root, dataset_name)
    elif dataset_name.lower() in ['reddit-binary']:
        transform = T.Constant(value=1.0)
        return TUDataset(dataset_root, dataset_name, transform=transform)
    else:
        raise ValueError(f"{dataset_name} is not defined.")