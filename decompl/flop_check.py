import sys
sys.path.append(".")
from train import *
from DECOMPL import *
import torchvision.transforms as T
from volleyball import *
from dataset import *


cfg = Config('volleyball')

cfg.image_size = 360, 640  # input image size
cfg.batch_size =  8  # train batch size 
cfg.test_batch_size = 1  # test batch size
cfg.train_random_seed = 0 # seed

cfg.use_multi_gpu = False # for nn data parallel
cfg.device_list = "0" # gpu device list
cfg.test_before_train = False

# optimizer settings
cfg.train_learning_rate = 1e-4
cfg.lr_plan = {30:5e-5, 60:2e-5, 90:1e-5}
cfg.max_epoch = 120
cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]  
cfg.exp_note = 'Volleyball'


from fvcore.nn import FlopCountAnalysis, flop_count_table


model = DECOMPL_volleyball(cfg)

# Reading dataset
training_set, validation_set = return_dataset(cfg)

# Create DataLoaders
params = {
    'batch_size': 1,
    'shuffle': True,
    'num_workers': 1,
    'pin_memory': True,
    'drop_last': False
}

validation_loader = data.DataLoader(validation_set, **params)

for i, batch_data_test in enumerate(validation_loader):
    if i % 100 == 0:
        print("test batch", i)

    # send the batch to CUDA
    images_in = batch_data_test["imgs"]
    boxes_in = batch_data_test["boxes"]
    flops = FlopCountAnalysis(model, [images_in, boxes_in])
    print(flop_count_table(flops))
    exit()

