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
config.lr = 0.01 # 0.1
config.momentum = 0.9
config.weight_decay = 5e-4 # 1e-4
config.loss = "cross_entropy"
# config.loss = "focal_loss"


# Train Dataset
config.dataset_name = "ms1mv3"
config.root_dir = "/home/nghia/dataset/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510

# Validation Dataset - LFW test pairs
config.val_check_interval = 0.05
config.bin_path = "/home/nghia/dataset/ms1m-retinaface-t1/lfw.bin"    
