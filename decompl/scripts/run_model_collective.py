import sys
sys.path.append(".")
from train import *

cfg = Config('collective')


cfg.image_size = 360, 640  # input image size
cfg.batch_size =  16  # train batch size 
cfg.test_batch_size = 4  # test batch size
cfg.train_random_seed = 0 # seed

cfg.num_boxes = 13
cfg.num_actions = 6
cfg.num_activities = 5

cfg.use_multi_gpu = False # for nn data parallel
cfg.device_list = "0,1" # gpu device list
cfg.test_only = True

# optimizer settings
cfg.train_learning_rate = 1e-5
cfg.lr_plan = {30:5e-6, 60:2e-6, 90:1e-6}
cfg.max_epoch = 120
cfg.exp_note = 'Collective'

# give checkpoint path or None
cfg.load_path = "./checkpoint_weights_collective_half.pth" # None 


train(cfg)
