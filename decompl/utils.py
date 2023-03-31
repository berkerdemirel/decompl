import torch
import time
import numpy as np
import torchvision.transforms as transforms
from thop import profile, clever_format
import torch.nn as nn
from config import *

from typing import Dict, Union


"""
Reference:
https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark/blob/4648310a42ca7b66013da9d623e9f856a483f30c/utils.py
"""

def prep_images(images: torch.Tensor) -> torch.Tensor:
    
    images = images.div(255.0)
    images = torch.sub(images,0.5)
    images = torch.mul(images,2.0)
    return images


def print_log(file_path: str, *args: str):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args,file=f)



def show_config(cfg: Config):
    print_log(cfg.log_path, '=====================Config=====================')
    for k, v in cfg.__dict__.items():
        print_log(cfg.log_path, k,': ',v)
    print_log(cfg.log_path, '======================End=======================')
    

def show_epoch_info(phase: str, log_path: str, info: Dict[str, Union[int, float, np.ndarray]]):
    print_log(log_path, '')
    if phase == 'Test':
        print_log(log_path, '====> %s at epoch #%d'%(phase, info['epoch']))
    else:
        print_log(log_path, '%s at epoch #%d'%(phase, info['epoch']))
        
    print_log(log_path, 'Group Activity Accuracy: %.2f%%, Actions Accuracy: %.2f%%, Activities Side Accuracy: %.2f%%, Activities Team Accuracy: %.2f%%, Loss: %.5f, BaselineLoss: %.5f, Using %.1f seconds'%(
                info['activities_acc'], info['actions_acc'], info['activities_acc_side'], info['activities_acc_team'], info['loss'], info['acties+acts'],  info['time']))

    if 'activities_conf' in info.keys():
        print_log(log_path, info['activities_conf'])
    if 'activities_MPCA' in info.keys():
        print_log(log_path, 'Activities MPCA:{:.2f}%'.format(info['activities_MPCA']))
    if 'MAD' in info.keys():
        print_log(log_path, 'MAD:{:.4f}'.format(info['MAD']))
    print_log(log_path, '\n')
        
    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class Timer(object):

    def __init__(self):
        self.last_time = time.time()
        
    def timeit(self) -> float:
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time-old_time


def MPCA(conf_mat: np.ndarray) -> np.ndarray:
    class_sum = np.sum(conf_mat, axis = 1, dtype = np.float32)
    for i in range(len(class_sum)):
        class_sum[i] = np.float32(conf_mat[i][i])/np.float32(class_sum[i])
    mpca = np.mean(class_sum)*100
    return mpca


class ConfusionMeter(object):
    def __init__(self, k: np.ndarray, normalized: bool=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted: torch.Tensor, target: torch.Tensor):
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)

        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self) -> np.ndarray:
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


if __name__ == "__main__":
    cfg = Config("volleyball")
    cfg.init_config()
    show_config(cfg)