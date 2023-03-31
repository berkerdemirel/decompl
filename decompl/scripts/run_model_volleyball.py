import sys
sys.path.append(".")
from train import *

cfg = Config('volleyball')


cfg.image_size = 360, 640  # input image size
cfg.batch_size =  16  # train batch size 
cfg.test_batch_size = 1  # test batch size
cfg.train_random_seed = 0 # seed

cfg.use_multi_gpu = False # for nn data parallel
cfg.device_list = "0,1" # gpu device list
cfg.test_only = True

# optimizer settings
cfg.train_learning_rate = 1e-4
cfg.lr_plan = {30:5e-5, 60:2e-5, 90:1e-5}
cfg.max_epoch = 120
cfg.actions_weights = [[1., 1., 2., 3., 1., 2., 2., 0.2, 1.]]  
cfg.exp_note = 'Volleyball'

# give checkpoint path or None
cfg.load_path = "./checkpoint_weights_volleyball_half.pth" # None 


train(cfg)
