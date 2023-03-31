import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms

import random
from PIL import Image
import numpy as np

from collections import Counter
from typing import Dict, Tuple, List, Union


"""
Reference:
https://github.com/cvlab-epfl/social-scene-understanding/blob/master/volleyball.py
https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark/blob/main/volleyball.py
"""

FRAMES_NUM = {1: 302, 2: 347, 3: 194, 4: 257, 5: 536, 6: 401, 7: 968, 8: 221, 9: 356, 10: 302, 
              11: 1813, 12: 1084, 13: 851, 14: 723, 15: 464, 16: 1021, 17: 905, 18: 600, 19: 203, 20: 342, 
              21: 650, 22: 361, 23: 311, 24: 321, 25: 617, 26: 734, 27: 1804, 28: 470, 29: 635, 30: 356, 
              31: 690, 32: 194, 33: 193, 34: 395, 35: 707, 36: 914, 37: 1049, 38: 653, 39: 518, 40: 401, 
              41: 707, 42: 420, 43: 410, 44: 356}

 
FRAMES_SIZE = {1: (480, 720), 2: (480, 720), 3: (480, 720), 4: (480, 720), 5: (480, 720), 6: (480, 720), 7: (480, 720), 8: (480, 720), 9: (480, 720), 10: (480, 720), 
               11: (480, 720), 12: (480, 720), 13: (480, 720), 14: (480, 720), 15: (450, 800), 16: (480, 720), 17: (480, 720), 18: (480, 720), 19: (480, 720), 20: (450, 800), 
               21: (450, 800), 22: (450, 800), 23: (450, 800), 24: (450, 800), 25: (480, 720), 26: (480, 720), 27: (480, 720), 28: (480, 720), 29: (480, 720), 30: (480, 720), 
               31: (480, 720), 32: (480, 720), 33: (480, 720), 34: (480, 720), 35: (480, 720), 36: (480, 720), 37: (480, 720), 38: (480, 720), 39: (480, 720), 40: (480, 720), 
               41: (480, 720), 42: (480, 720), 43: (480, 720), 44: (480, 720)}


ACTIONS = ['NA','Crossing','Waiting','Queueing','Walking','Talking']
ACTIVITIES = ['Crossing','Waiting','Queueing','Walking','Talking']

ACTIONS_ID = {a:i for i,a in enumerate(ACTIONS)}
ACTIVITIES_ID = {a:i for i,a in enumerate(ACTIVITIES)}
Action6to5 = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
Activity5to4 = {0:0, 1:1, 2:2, 3:0, 4:3}


def collective_read_annotations(path: str, sid: int) -> Dict[int, Dict[str, Union[int, List[int], np.ndarray]]]:
    annotations={}
    path=path + '/seq%02d/annotations.txt' % sid
    
    with open(path,mode='r') as f:
        frame_id=None
        group_activity=None
        actions=[]
        bboxes=[]
        for l in f.readlines():
            values=l[:-1].split('	')
            if int(values[0])!=frame_id:
                if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
                    counter = Counter(actions).most_common(2)
                    group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
                    annotations[frame_id]={
                        'frame_id':frame_id,
                        'group_activity':group_activity,
                        'actions':actions,
                        'bboxes':bboxes
                    }
                    
                frame_id=int(values[0])
                group_activity=None
                actions=[]
                bboxes=[]
                
            actions.append(int(values[5])-1)
            x,y,w,h = (int(values[i])  for i  in range(1,5))
            H,W=FRAMES_SIZE[sid]
            
            bboxes.append( (y/H,x/W,(y+h)/H,(x+w)/W) )
        
        if frame_id != None and frame_id % 10 == 1 and frame_id + 9 <= FRAMES_NUM[sid]:
            counter = Counter(actions).most_common(2)
            group_activity= counter[0][0] - 1 if counter[0][0] != 0 else counter[1][0] - 1
            annotations[frame_id]={
                'frame_id':frame_id,
                'group_activity':group_activity,
                'actions':actions,
                'bboxes':bboxes
            }
    return annotations
            
        
def collective_read_dataset(path: int, seqs: List[int]) -> List[Dict[int, Dict[str, Union[int, List[int], np.ndarray]]]]:
    data = {}
    for sid in seqs:
        data[sid] = collective_read_annotations(path,sid)
    return data


def collective_all_frames(anns: List[Dict[int, Dict[str, Union[int, List[int], np.ndarray]]]]) -> List[Tuple[int, int]]:
    return [(s,f)  for s in anns for f in anns[s] ]


class CollectiveDataset(data.Dataset):
    def __init__(self, 
                 anns: Dict[int, Dict[int, Tuple[str, int, List[int], np.ndarray]]], 
                 frames: List[Tuple[int, int]], 
                 images_path: str, 
                 image_size: Tuple[int, int], 
                 feature_size: Tuple[int, int], 
                 num_boxes: int=13, 
                 num_frames: int=10, 
                 is_training: bool=True, 
                 flip: bool=False):
        
        self.anns=anns
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.feature_size=feature_size
        
        self.num_boxes = num_boxes
        self.num_frames = num_frames
        
        self.is_training=is_training
        self.flip = flip

    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
        select_frames=self.get_frames(self.frames[index])
        sample=self.load_samples_sequence(select_frames)
        return sample
    
    def get_frames(self, frame: Tuple[int, int]) -> List[Tuple[int, int, int]]:
        sid, src_fid = frame
        if self.is_training:
            fid = random.randint(src_fid, src_fid+self.num_frames-1)
            return [(sid, src_fid, fid)]
        else:
            return [(sid, src_fid, fid) 
                    for fid in range(src_fid, src_fid+self.num_frames)]
            
    def load_samples_sequence(self, select_frames: List[Tuple[int, int, int]]) -> Dict[str, Union[str, torch.Tensor]]:
        OH, OW = self.feature_size
        
        images, bboxes = [], []
        activities, actions, pths = [], [], []
        bboxes_num=[]
        rnd = np.random.rand() # draw sample from uniform distribution
        
        for i, (sid, src_fid, fid) in enumerate(select_frames):
            img = Image.open(self.images_path + '/seq%02d/frame%04d.jpg'%(sid,fid))
            img = transforms.functional.resize(img,self.image_size)
            img = np.array(img)
            pth = self.images_path + '/seq%02d/frame%04d.jpg' % (sid, fid)

            # H,W,3 -> 3,H,W
            img = img.transpose(2,0,1)
            images.append(img)
            
            temp_boxes = []
            for box in self.anns[sid][src_fid]['bboxes']:
                y1, x1, y2, x2 = box
                if self.flip and rnd < 0.5:
                    w1, h1, w2, h2 = (1-x2)*OW, y1*OH, (1-x1)*OW, y2*OH  
                else:
                    w1, h1, w2, h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes.append((w1, h1, w2, h2))
                
            if self.flip and rnd < 0.5: # flip
                img = np.flip(img, axis=2)

            temp_actions = [Action6to5[i] for i in self.anns[sid][src_fid]['actions'][:]]
            bboxes_num.append(len(temp_boxes))
            
            while len(temp_boxes) != self.num_boxes:
                temp_boxes.append((0,0,0,0))
                temp_actions.append(-1)

            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            pths.append(pth)
            activities.append(Activity5to4[self.anns[sid][src_fid]['group_activity']])
        
        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes = np.array(bboxes,dtype=np.float).reshape(-1, self.num_boxes, 4)
        actions = np.array(actions,dtype=np.int32).reshape(-1, self.num_boxes)
        
        #convert to pytorch tensor
        images = torch.from_numpy(images).float()
        bboxes = torch.from_numpy(bboxes).float()
        actions = torch.from_numpy(actions).long()
        activities = torch.from_numpy(activities).long()
        bboxes_num = torch.from_numpy(bboxes_num).int()
        
        sample = {"imgs": images, "boxes": bboxes, "actions": actions, "activities": activities, "bboxes_num": bboxes_num, "paths": pths}
        return sample