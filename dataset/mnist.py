import torchvision
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2

class MNISTTrain(torchvision.datasets.MNIST):
    def __init__(self, root: str, *args, **kwargs) -> None:
        self.transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        super().__init__(root, True, self.transforms, download=True)
    
    def __getitem__(self, index: int):
        img, label = super().__getitem__(index)
        img = img.repeat(3, 1, 1)
        return img, label

class MNISTVal(torchvision.datasets.MNIST):
    def __init__(self, root: str) -> None:
        self.transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        super().__init__(root, False, self.transforms, download=True)


    def __getitem__(self, index: int):
        img1, label1 = super().__getitem__(index)
        img1 = img1.repeat(3, 1, 1)
        if index == super().__len__() - 1:
            index2 = 0
        else: index2 = index + 1
        img2, label2 = super().__getitem__(index2)
        img2 = img2.repeat(3, 1, 1)
        is_same = int(label1 == label2)
        return (img1, img2) ,is_same
    
    
# for debug
if __name__ == "__main__":
    import torchvision.transforms.functional as F

    dataset = MNISTTrain("/home/nghia/dataset/mnist")
    val_dataset = MNISTVal("/home/nghia/dataset/mnist")
    val_dataset[0]
    indices = np.arange(stop=len(dataset))
    np.random.shuffle(indices)
    for idx in indices:
        img, label = dataset[idx]
        img = np.asarray(F.to_pil_image(((img * 0.5 + 0.5)*255).to(torch.uint8)))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("pair", img)
        print(label)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break
    print("end")
