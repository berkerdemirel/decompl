import time
import os
from dataclasses import dataclass

@dataclass
class Config(object):
    def __init__(self, dataset_name: str):
        # Global
        self.image_size = None
        self.batch_size =  None 
        self.test_batch_size = None
        self.num_boxes = 12 # max number of bounding boxes in each frame
        
        # VGG16
        self.backbone = 'vgg16'
        self.emb_features = 512

        # Gpu
        self.use_gpu = True
        self.use_multi_gpu = True   
        self.device_list = "0,1,2,3" # id list of gpus used for training 
        
        # Dataset
        assert(dataset_name in ['volleyball'], ['collective'])
        self.dataset_name = dataset_name 
        if dataset_name == 'volleyball':
            self.out_size = 22, 40
            self.data_path = '../data/volleyball_videos' # data path for the volleyball dataset
            self.train_seqs = [ 1,3,6,7,10,13,15,16,18,22,23,31,32,36,38,39,40,41,42,48,50,52,53,54,
                                0,2,8,12,17,19,24,26,27,28,30,33,46,49,51] # video id list of train set 
            self.test_seqs = [4,5,9,11,14,20,21,25,29,34,35,37,43,44,45,47] # video id list of test set
        elif dataset_name == 'collective':
            self.out_size = 57, 87
            self.data_path = '../data/collective'
            self.test_seqs=[5,6,7,8,9,10,11,15,16,25,28,29]
            self.train_seqs=[s for s in range(1,45) if s not in self.test_seqs]  
        
        # Backbone 
        self.crop_size = 4, 4 # crop size of roi align
        
        # Activity Action
        self.num_actions = 9 # number of action categories
        self.num_activities = 8 # number of activity categories
        self.actions_loss_weight = 1.0 # weight used to balance action loss and activity loss
        self.actions_weights = None

        # Sample
        self.num_frames = 1
        self.num_before = 5
        self.num_after = 4

        # Embedding dimension for boxes
        self.num_features_boxes = 128

        # Training Parameters
        self.train_random_seed = None
        self.train_learning_rate = 1e-4 # initial learning rate
        self.train_dropout_prob = 0.3 # dropout probability
        self.max_epoch = 120 # max training epoch
        
        # Exp
        self.test_before_train = False
        self.exp_note = 'Group-Activity-Recognition'
        self.exp_name = None
        self.load_path = None

        
    def init_config(self, need_new_folder: bool=True):
        if self.exp_name is None:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name = '[%s]<%s>'%(self.exp_note, time_str)
            
        self.result_path = 'result/%s'%self.exp_name
        self.log_path = 'result/%s/log.txt'%self.exp_name
            
        if need_new_folder:
            os.mkdir(self.result_path)


if __name__ == "__main__":
    cfg = Config("volleyball")
    cfg.init_config()
    print(cfg.exp_note)