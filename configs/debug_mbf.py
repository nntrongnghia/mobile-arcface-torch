from backbones.mobilefacenet import MobileFaceNet
from easydict import EasyDict
    
config = EasyDict()
# General
config.name = "debug_mbf"

# Model
config.embedding_size = 3
config.model = MobileFaceNet(num_features=config.embedding_size)

# Trainer
config.batch_size = 128
config.max_epochs = 20

# Optimizer
config.optimizer = "sgd"
config.lr = 0.01
config.momentum = 0.9
config.weight_decay = 5e-4

# Train Dataset
config.dataset_name = "mnist"
config.root = "/home/nghia/dataset/mnist"
config.num_classes = 10
config.val_check_interval = 1.0