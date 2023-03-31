import numpy as np
import skimage.io
import skimage.transform
import torch
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models
from torchvision.utils import save_image

from PIL import Image
import random

import sys
from typing import Dict, Tuple, List, Union

"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark/blob/main/volleyball.py
"""

ACTIVITIES = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
              'l_set', 'l-spike', 'l-pass', 'l_winpoint']

NUM_ACTIVITIES = 8

ACTIONS = ['blocking', 'digging', 'falling', 'jumping',
           'moving', 'setting', 'spiking', 'standing',
           'waiting']
NUM_ACTIONS = 9


def volley_read_annotations(path: str) -> Dict[int, Dict[str, Union[int, List[int], np.ndarray]]]:
    annotations = {}

    gact_to_id = {name: i for i, name in enumerate(ACTIVITIES)}
    act_to_id = {name: i for i, name in enumerate(ACTIONS)}

    with open(path) as f:
        for l in f.readlines():
            values = l[:-1].split(' ')
            file_name = values[0]
            # print(path)
            activity = gact_to_id[values[1]]

            values = values[2:]
            num_people = len(values) // 5

            action_names = values[4::5]
            actions = [act_to_id[name]
                       for name in action_names]

            def _read_bbox(xywh):
                x, y, w, h = map(int, xywh)
                return y, x, y+h, x+w
            bboxes = np.array([_read_bbox(values[i:i+4])
                               for i in range(0, 5*num_people, 5)])

            fid = int(file_name.split('.')[0])
            annotations[fid] = {
                'file_name': file_name,
                'group_activity': activity,
                'actions': actions,
                'bboxes': bboxes,
            }
    return annotations


def volley_read_dataset(path: str, seqs: List[int]) -> Dict[int, Dict[int, Tuple[str, int, List[int], np.ndarray]]]:
    data = {}
    for sid in seqs:
        # data[sid] = volley_read_annotations(path + '/%d/annotations.txt' % sid)
        data[sid] = volley_read_annotations('../reannotations/%d/annotations_corrected.txt' % sid)
    return data


def volley_all_frames(data: Dict[int, Dict[int, Tuple[str, int, List[int], np.ndarray]]]) -> List[Tuple[int, int]]:
    frames = []
    for sid, anns in data.items():
        for fid, _ in anns.items():
            frames.append((sid, fid))
    return frames


class VolleyballDataset(data.Dataset):
    def __init__(self, 
                anns: Dict[int, Dict[int, Tuple[str, int, List[int], np.ndarray]]], 
                tracks: Dict[Tuple[int, int], Dict[int, np.ndarray]], 
                frames: List[Tuple[int, int]], 
                images_path: str, 
                image_size: Tuple[int, int], 
                feature_size: Tuple[int, int], 
                num_boxes: int=12, 
                num_before: int=4, 
                num_after: int=4, 
                is_training: bool=True, 
                flip: bool=False):
        
        self.anns = anns
        self.tracks = tracks
        self.frames = frames
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size
        
        self.num_boxes = num_boxes
        self.num_before = num_before
        self.num_after = num_after
        
        self.is_training = is_training
        self.flip = flip

    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self,index: int) -> Dict[str, Union[str, torch.Tensor]]:
        select_frames = self.volley_frames_sample(self.frames[index])
        sample = self.load_samples_sequence(select_frames)
        return sample
    
    def volley_frames_sample(self, frame: Tuple[int, int]) -> List[Tuple[int, int, int]]:
        sid, src_fid = frame
        
        if self.is_training:
            fid = random.randint(src_fid-self.num_before, src_fid+self.num_after)
            return [(sid, src_fid, fid)]
        else:
            return [(sid, src_fid, fid)
                    for fid in range(src_fid-self.num_before, src_fid+self.num_after+1)]
       

    def load_samples_sequence(self, select_frames: List[Tuple[int, int, int]]) -> Dict[str, Union[str, torch.Tensor]]:
        OH, OW = self.feature_size
        images, boxes = [], []
        activities, actions, pths = [], [], []
        rnd = np.random.rand() # draw sample from uniform distribution

        for i, (sid, src_fid, fid) in enumerate(select_frames):
            img = Image.open(self.images_path + '/%d/%d/%d.jpg' % (sid, src_fid, fid))
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)

            # H,W,3 -> 3,H,W
            img = img.transpose(2,0,1)
            pth = self.images_path + '/%d/%d/%d.npy' % (sid, src_fid, fid)

            temp_boxes = np.ones_like(self.tracks[(sid, src_fid)][fid])
            act = self.anns[sid][src_fid]['actions']
            acty = self.anns[sid][src_fid]['group_activity']

            for i, track in enumerate(self.tracks[(sid, src_fid)][fid]):
                y1, x1, y2, x2 = track
                if self.flip and rnd < 0.5: # flip
                    w1, h1, w2, h2 = (1-x2)*OW, y1*OH, (1-x1)*OW, y2*OH  
                else: # do not flip
                    w1, h1, w2, h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes[i] = np.array([w1, h1, w2, h2])
            
            if self.flip and rnd < 0.5: # flip
                img = np.flip(img, axis=2)
                acty = (acty+4)%8
            
            images.append(img)
            boxes.append(temp_boxes)
            actions.append(act)
            activities.append(acty)
            pths.append(pth)

            if len(boxes[-1]) != self.num_boxes:
                boxes[-1] = np.vstack([boxes[-1], boxes[-1][:self.num_boxes-len(boxes[-1])]])
                actions[-1] = np.concatenate((actions[-1], [actions[-1][-1]]*(self.num_boxes-len(actions[-1]))))

        images = np.stack(images)
        bboxes = np.vstack(boxes).reshape([-1, self.num_boxes, 4])
        actions = np.hstack(actions).reshape([-1, self.num_boxes])
        activities = np.array(activities, dtype=np.int32)

        #convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()

        sample = {"imgs": images, "boxes":bboxes, "actions": actions, "activities":activities, "paths":pths}
        
        return sample
    