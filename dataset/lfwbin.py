# https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface
# https://drive.google.com/file/d/1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy/view
# Dataset for lfw.bin
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import logging
from joblib import Parallel, delayed
from pqdm.threads import pqdm
import time
from tqdm import tqdm
from torchvision.io import decode_image


class LFWBin(Dataset):
    def __init__(self, bin_path: str) -> None:
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        with open(bin_path, "rb") as f:
            self.bins, self.labels = pickle.load(f, encoding="bytes")
        self.image1s = []
        self.image2s = []
        for i, b in tqdm(enumerate(self.bins)):
            img = decode_image(torch.tensor(b.reshape(-1)))
            img = self.transform(img)
            if i % 2 == 0:
                self.image1s.append(img)
            else:
                self.image2s.append(img)
        self.labels = torch.tensor(self.labels)
        self.image1s = torch.stack(self.image1s)
        self.image2s = torch.stack(self.image2s)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.image1s[index], self.image2s[index]), self.labels[index]


# for debug
if __name__ == "__main__":
    import torchvision.transforms.functional as F

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    dataset = LFWBin("/home/nghia/dataset/ms1m-retinaface-t1/lfw.bin")
    print(dataset.labels)
    print("hold")
    loader = DataLoader(dataset, 10)
    # for imgs, labels in loader:
    #     print(labels)
    indices = np.arange(stop=len(dataset))
    np.random.shuffle(indices)
    for idx in indices:
        (img1, img2), label = dataset[idx]
        img1 = cv2.cvtColor(
            np.asarray(F.to_pil_image(((img1.detach() * 0.5 + 0.5) * 255).to(torch.uint8))), cv2.COLOR_RGB2BGR
        )
        img2 = cv2.cvtColor(
            np.asarray(F.to_pil_image(((img2.detach() * 0.5 + 0.5) * 255).to(torch.uint8))), cv2.COLOR_RGB2BGR
        )
        # img1 = np.asarray((img1.detach() * 0.5 + 0.5) * 255).astype(np.uint8)
        # img2 = np.asarray((img2.detach() * 0.5 + 0.5) * 255).astype(np.uint8)
        cv2.imshow("pair", cv2.hconcat([img1, img2]))
        print(label)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break
    print("end")
