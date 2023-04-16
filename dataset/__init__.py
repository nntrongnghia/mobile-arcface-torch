from .ms1mv3 import MS1MV3
from .mnist import MNISTTrain
from torch.utils.data import DataLoader, Dataset

def get_train_dataset(dataset_name: str, **kwargs) -> Dataset:
    if dataset_name == "ms1mv3":
        return MS1MV3(**kwargs)
    if dataset_name == "mnist":
        return MNISTTrain(**kwargs)
    else: 
        raise ValueError("Dataset name not defined")