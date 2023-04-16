from backbones.mobilefacenet import MobileFaceNet
from easydict import EasyDict
    
config = EasyDict()
# General
config.name = "ms1mv3_arcface_mbf"

# Model
config.model = MobileFaceNet()
config.embedding_size = 512

# Trainer
config.batch_size = 128
config.max_epochs = 20

# Optimizer
config.optimizer = "sgd"
config.lr = 0.01
config.momentum = 0.9
config.weight_decay = 5e-4

# config.optimizer = "adam"
# config.lr = 0.001
# config.weight_decay = 5e-4


# Train Dataset
config.dataset_name = "ms1mv3"
config.root_dir = "/home/nghia/dataset/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510

# Test Dataset - LFW Testset
config.lfwpair_kwargs = {
    "img_dir": "/home/nghia/dataset/LFW/lfw-deepfunneled",
    "pairs_txt_path": "/home/nghia/dataset/LFW/pairs.txt"
}
config.val_check_interval = 0.05
    