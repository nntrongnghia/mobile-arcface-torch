# Download the Deep funneling LFW dataset here: vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
import os
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
# Use the pairs.txt described in http://vis-www.cs.umass.edu/lfw/#views
# This dataset is to test the 1:1 verification
class LFWPair(Dataset):
    def __init__(self, img_dir: str, pairs_txt_path: str, apply_normalize=True) -> None:
        super().__init__()
        tf = [transforms.ToTensor(), transforms.Resize((112, 112))] 
        if apply_normalize:
            tf.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transform = transforms.Compose(tf)
        self.img_dir = img_dir
        with open(pairs_txt_path) as f:
            lines = f.readlines()[1:]
            self.pairs = [l.split() for l in lines]
        self.samples = None
        self.image1s = None
        self.image2s = None
        self.labels = None

    def load_all(self):
        logging.info("Loading all samples of LFW Test Pairs ...")
        t1 = time.time()
        self.samples = pqdm(range(len(self)), self.__getitem__, n_jobs=6)
        self.image1s = torch.stack([s[0][0] for s in self.samples])
        self.image2s = torch.stack([s[0][1] for s in self.samples])
        self.labels = torch.stack([s[1] for s in self.samples])
        print(self.image1s.shape)
        print(self.image2s.shape)
        print(self.labels.shape)
        t2 = time.time()
        logging.info(f"Done. Load all samples in {t2 - t1:.1f}s")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # Please refer to LFW README
        pair = self.pairs[index]

        def get_image(name, img_id):
            path = os.path.join(self.img_dir, name, f"{name}_{img_id:0>4}.jpg")
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(pair) == 3:  # same person match
            img1 = get_image(pair[0], pair[1])
            img2 = get_image(pair[0], pair[2])
            label = 1  # True == same person
        elif len(pair) == 4:  # person mismatch
            img1 = get_image(pair[0], pair[1])
            img2 = get_image(pair[2], pair[3])
            label = 0
        label = torch.tensor(label)
        if self.transform:
            imgs = [self.transform(im) for im in [img1, img2]]
        else:
            imgs = [img1, img2]
        return imgs, label


# for debug
if __name__ == "__main__":
    import torchvision.transforms.functional as F
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    dataset = LFWPair(
        "/home/nghia/dataset/LFW/lfw-deepfunneled",
        "/home/nghia/dataset/LFW/pairs.txt",
        False,
    )
    dataset.load_all()
    print(dataset.labels)
    print("hold")
    # loader = DataLoader(dataset, 10)
    # for imgs, labels in loader:
    #     print(labels)
    # indices = np.arange(stop=len(dataset))
    # np.random.shuffle(indices)
    # for idx in indices:
    #     (img1, img2), label = dataset[idx]
    #     img1 = cv2.cvtColor(
    #         np.asarray(F.to_pil_image(img1.detach())), cv2.COLOR_RGB2BGR
    #     )
    #     img2 = cv2.cvtColor(
    #         np.asarray(F.to_pil_image(img2.detach())), cv2.COLOR_RGB2BGR
    #     )
    #     cv2.imshow("pair", cv2.hconcat([img1, img2]))
    #     print(label)
    #     key = cv2.waitKey(0)
    #     if key == 27:  # ESC to exit
    #         break
    # print("end")
