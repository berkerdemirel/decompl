import torch
import torch.optim as optim

import time
import random
import os
import sys

from config import *
from volleyball import *
from dataset import *
from DECOMPL import *
from utils import *
from typing import Dict, Union


# for reproducibility
def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


# learning rate update
def adjust_lr(optimizer: torch.optim.Optimizer, new_lr: float):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train(cfg: Config):

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)

    # Assign seeds
    set_seeds(cfg.train_random_seed)
    
    # Reading dataset
    training_set, validation_set = return_dataset(cfg)
    
    # Create DataLoaders
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 1,
        'pin_memory': True,
        'drop_last': True
    }
    
    training_loader = data.DataLoader(training_set, **params)
    
    params['batch_size'] = cfg.test_batch_size
    params['drop_last'] = False
    validation_loader = data.DataLoader(validation_set, **params)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    model_list = {'volleyball':DECOMPL_volleyball, 'collective':DECOMPL_collective}
    model = model_list[cfg.dataset_name](cfg)

    if cfg.load_path != None: # load pretrained model
        model.load_state_dict(torch.load(cfg.load_path, map_location=device)) 
    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)
        model = model.to(device=device)
    else:
        model = model.to(device=device)
    
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate)
    
    # Choose corresponding functions for training and evaluation
    train_list = {'volleyball':train_volleyball, 'collective':train_collective}
    test_list = {'volleyball':test_volleyball, 'collective':test_collective}
    train = train_list[cfg.dataset_name]
    test = test_list[cfg.dataset_name]
    
    if cfg.test_only:
        test_info = test(validation_loader, model, device, 0, cfg)
        print(test_info)
        exit()

    # Training iteration
    best_result = {'epoch':0, 'activities_acc':0}
    start_epoch = 1


    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])

        # train one epoch
        train_info = train(training_loader, model, device, optimizer, epoch, cfg)
        # info
        show_epoch_info('Train', cfg.log_path, train_info)

        # test
        test_info = test(validation_loader, model, device, epoch, cfg)
        # info
        show_epoch_info('Test', cfg.log_path, test_info)
            
        # update best result and save
        if test_info['activities_acc'] >= best_result['activities_acc']:
            best_result = test_info
            filepath = cfg.result_path+'/epoch%d_%.2f%%.pth'%(epoch, test_info['activities_acc'])
            torch.save(model.state_dict(), filepath)
            print('model saved to:', filepath)
        
        print_log(cfg.log_path, 
                    'Best group activity accuracy: %.2f%% at epoch #%d.'%(best_result['activities_acc'], best_result['epoch']))
        
   

def train_volleyball(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device, optimizer: torch.optim.Optimizer, epoch: int, cfg: Config)\
     -> Dict[str, Union[int, float, np.ndarray]]:
    model.train()
    # Accumulate accuracies and losses
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    activities_meter_side = AverageMeter()
    activities_meter_team = AverageMeter()
    loss_meter = AverageMeter()
    acties_acts_meter = AverageMeter()
    # Timer
    epoch_timer = Timer()
    # Confusion matrix
    activities_conf = ConfusionMeter(cfg.num_activities)

    for i, batch_data in enumerate(data_loader):
        if i % 100 == 0:
            print("train batch", i)
        # send the batch to CUDA
        images_in = batch_data["imgs"].to(device=device)
        boxes_in = batch_data["boxes"].to(device=device)
        actions_in = batch_data["actions"].to(device=device)
        activities_in = batch_data["activities"].to(device=device)
        pth = np.array(batch_data["paths"])
        
        batch_size = images_in.shape[0]
        num_frames = images_in.shape[1]

        # prepare labels
        actions_in = actions_in.reshape((batch_size, num_frames, cfg.num_boxes))
        activities_in = activities_in.reshape((batch_size, num_frames))

        actions_in = actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
        activities_in = activities_in[:,0].reshape((batch_size,))

        # prepare team labels
        activities_in_team = activities_in.detach().clone().fmod(4)

        # prepare side labels
        activities_in_side = activities_in.detach().clone() 
        activities_in_side[activities_in >= 4] = 1
        activities_in_side[activities_in < 4] = 0

        # forward
        actions_scores, activities_scores_side, activities_scores_team, activities_scores = model((images_in, boxes_in))
        # activities loss
        activities_loss = F.cross_entropy(activities_scores, activities_in)
        # actions loss
        actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
        actions_loss = F.cross_entropy(actions_scores,actions_in,weight=actions_weights[0])  
        # side and team loss
        activities_loss_side = F.cross_entropy(activities_scores_side, activities_in_side)
        activities_loss_team = F.cross_entropy(activities_scores_team, activities_in_team)
        
        # predict labels
        activities_labels = torch.argmax(activities_scores, dim=1)
        actions_labels = torch.argmax(actions_scores, dim=1)  
        activities_labels_side = torch.argmax(activities_scores_side, dim=1)
        activities_labels_team = torch.argmax(activities_scores_team, dim=1)

        # count correct activities
        activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
        # count correct actions
        actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
        # count correct teams and sides
        activities_correct_side = torch.sum(torch.eq(activities_labels_side.int(), activities_in_side.int()).float())
        activities_correct_team = torch.sum(torch.eq(activities_labels_team.int(), activities_in_team.int()).float())

        # Get accuracies
        activities_accuracy = activities_correct.item() / activities_scores.shape[0]
        actions_accuracy = actions_correct.item() / actions_scores.shape[0]
        activities_accuracy_side = activities_correct_side.item() / activities_scores_side.shape[0]
        activities_accuracy_team = activities_correct_team.item() / activities_scores_team.shape[0]

        # Total loss
        total_loss = activities_loss + activities_loss_team + activities_loss_side + actions_loss

        # update meters
        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy, activities_scores.shape[0])
        activities_meter_side.update(activities_accuracy_side, activities_scores_side.shape[0])
        activities_meter_team.update(activities_accuracy_team, activities_scores_team.shape[0])
        loss_meter.update(total_loss.item(), batch_size)
        acties_acts_meter.update(activities_loss.item() + actions_loss.item(), batch_size)
        activities_conf.add(activities_labels, activities_in)

        # step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'acties+acts': acties_acts_meter.avg,
        'activities_acc': activities_meter.avg*100,
        'actions_acc': actions_meter.avg*100,
        'activities_acc_side': activities_meter_side.avg*100,
        'activities_acc_team': activities_meter_team.avg*100,
        'activities_conf': activities_conf.value()
    }
    
    return train_info
        
    
def test_volleyball(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device, epoch: int, cfg: Config) \
    -> Dict[str, Union[int, float, np.ndarray]]:
    
    model.eval()

    # Accumulate accuracies and losses
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    activities_meter_side = AverageMeter()
    activities_meter_team = AverageMeter()
    acties_acts_meter = AverageMeter()
    loss_meter = AverageMeter()

    # Timer
    epoch_timer = Timer()
    # Confusion Matrix
    activities_conf = ConfusionMeter(cfg.num_activities)

    with torch.inference_mode():
        for i, batch_data_test in enumerate(data_loader):
            if i % 100 == 0:
                print("test batch", i)

            # send the batch to CUDA
            images_in = batch_data_test["imgs"].to(device=device)
            boxes_in = batch_data_test["boxes"].to(device=device)
            actions_in = batch_data_test["actions"].to(device=device)
            activities_in = batch_data_test["activities"].to(device=device)
            pth = np.array(batch_data_test["paths"])
            
            batch_size = images_in.shape[0]
            num_frames = images_in.shape[1]

            # prepare labels
            actions_in = actions_in.reshape((batch_size, num_frames, cfg.num_boxes))
            activities_in = activities_in.reshape((batch_size, num_frames))
            
            actions_in = actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            activities_in = activities_in[:,0].reshape((batch_size,))

            # prepare team labels
            activities_in_team = activities_in.detach().clone().fmod(4)

            # prepare side labels
            activities_in_side = activities_in.detach().clone() 
            activities_in_side[activities_in >= 4] = 1
            activities_in_side[activities_in < 4] = 0

            # forward
            actions_scores, activities_scores_side, activities_scores_team, activities_scores = model((images_in, boxes_in))


            # activities loss
            activities_loss = F.cross_entropy(activities_scores, activities_in)
            # actions loss
            actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
            actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights[0])  
            # side and team loss
            activities_loss_side = F.cross_entropy(activities_scores_side, activities_in_side)
            activities_loss_team = F.cross_entropy(activities_scores_team, activities_in_team)

            # predict labels
            activities_labels = torch.argmax(activities_scores, dim=1)
            actions_labels = torch.argmax(actions_scores, dim=1)  
            activities_labels_side = torch.argmax(activities_scores_side, dim=1)
            activities_labels_team = torch.argmax(activities_scores_team, dim=1)

            # count correct activities
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
            # count correct actions
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            # count correct teams and sides
            activities_correct_side = torch.sum(torch.eq(activities_labels_side.int(), activities_in_side.int()).float())
            activities_correct_team = torch.sum(torch.eq(activities_labels_team.int(), activities_in_team.int()).float())

            # Get accuracies
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            activities_accuracy_side = activities_correct_side.item() / activities_scores_side.shape[0]
            activities_accuracy_team = activities_correct_team.item() / activities_scores_team.shape[0]

            # Total loss
            total_loss = activities_loss + activities_loss_team + activities_loss_side + actions_loss

            # update meters
            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])
            activities_meter_side.update(activities_accuracy_side, activities_scores_side.shape[0])
            activities_meter_team.update(activities_accuracy_team, activities_scores_team.shape[0])
            loss_meter.update(total_loss.item(), batch_size)
            acties_acts_meter.update(activities_loss.item() + actions_loss.item(), batch_size)
            activities_conf.add(activities_labels, activities_in)
            
    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'acties+acts': acties_acts_meter.avg,
        'activities_acc': activities_meter.avg*100,
        'actions_acc': actions_meter.avg*100,
        'activities_acc_side': activities_meter_side.avg*100,
        'activities_acc_team': activities_meter_team.avg*100,
        'activities_conf': activities_conf.value()
    }
    
    return test_info



def train_collective(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device, optimizer: torch.optim.Optimizer, epoch: int, cfg: Config)\
     -> Dict[str, Union[int, float, np.ndarray]]:
    model.train()
    # Accumulate accuracies and losses
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    activities_meter_side = AverageMeter()
    activities_meter_team = AverageMeter()
    loss_meter = AverageMeter()
    acties_acts_meter = AverageMeter()
    # Timer
    epoch_timer = Timer()
    # Confusion matrix
    activities_conf = ConfusionMeter(cfg.num_activities)
    for i, batch_data in enumerate(data_loader):
        if i % 100 == 0:
            print("train batch", i)
        # send the batch to CUDA
        images_in = batch_data["imgs"].to(device=device)
        boxes_in = batch_data["boxes"].to(device=device)
        actions_in = batch_data["actions"].to(device=device)
        activities_in = batch_data["activities"].to(device=device)
        bboxes_num = batch_data["bboxes_num"].to(device=device)

        pth = np.array(batch_data["paths"])
        
        batch_size = images_in.shape[0]
        num_frames = images_in.shape[1]

        # prepare labels
        actions_in = actions_in.reshape((batch_size, num_frames, cfg.num_boxes))
        # activities_in = activities_in.reshape((batch_size,))
        activities_in = activities_in.reshape((batch_size, num_frames))
        activities_in = activities_in[:,0].reshape((batch_size,))

        actions_in_nopad = []
        actions_in = actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
        bboxes_num = bboxes_num.reshape(batch_size*num_frames,)
        for bt in range(batch_size*num_frames):
            N = bboxes_num[bt]
            actions_in_nopad.append(actions_in[bt,:N])
        actions_in = torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
        bboxes_num_in = bboxes_num.reshape(batch_size, num_frames)

        # forward
        actions_scores, activities_scores = model((images_in, boxes_in, bboxes_num_in))

        # activities loss
        activities_loss = F.cross_entropy(activities_scores, activities_in)
        # actions loss
        actions_loss = F.cross_entropy(actions_scores,actions_in,weight=None)  
        
        # predict labels
        activities_labels = torch.argmax(activities_scores, dim=1)
        actions_labels = torch.argmax(actions_scores, dim=1)  

        # count correct activities
        activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
        # count correct actions
        actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())

        # Get accuracies
        activities_accuracy = activities_correct.item() / activities_scores.shape[0]
        actions_accuracy = actions_correct.item() / actions_scores.shape[0]

        # Total loss
        total_loss = activities_loss + actions_loss

        # update meters
        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy, activities_scores.shape[0])
        loss_meter.update(total_loss.item(), batch_size)
        activities_conf.add(activities_labels, activities_in)

        # step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'acties+acts': acties_acts_meter.avg,
        'activities_acc': activities_meter.avg*100,
        'actions_acc': actions_meter.avg*100,
        'activities_acc_side': activities_meter_side.avg*100,
        'activities_acc_team': activities_meter_team.avg*100,
        'activities_conf': activities_conf.value()
    }
    return train_info
        
    
def test_collective(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device, epoch: int, cfg: Config) \
    -> Dict[str, Union[int, float, np.ndarray]]:
    model.eval()
    # Accumulate accuracies and losses
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    activities_meter_side = AverageMeter()
    activities_meter_team = AverageMeter()
    acties_acts_meter = AverageMeter()
    loss_meter = AverageMeter()
    # Timer
    epoch_timer = Timer()
    # Confusion Matrix
    activities_conf = ConfusionMeter(cfg.num_activities)

    with torch.inference_mode():
        for i, batch_data_test in enumerate(data_loader):
            if i % 100 == 0:
                print("test batch", i)

            # send the batch to CUDA
            images_in = batch_data_test["imgs"].to(device=device)
            boxes_in = batch_data_test["boxes"].to(device=device)

            actions_in = batch_data_test["actions"].to(device=device)
            activities_in = batch_data_test["activities"].to(device=device)
            bboxes_num = batch_data_test["bboxes_num"].to(device=device)
            pth = np.array(batch_data_test["paths"])
            
            batch_size = images_in.shape[0]
            num_frames = images_in.shape[1]
            bboxes_num = bboxes_num.reshape(batch_size,num_frames)

            # prepare labels
            actions_in_nopad = []
            actions_in = actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
            bboxes_num = bboxes_num.reshape(batch_size*num_frames,)
            # actions_in 
            for bt in range(0, batch_size*num_frames, num_frames):
                N = bboxes_num[bt]
                actions_in_nopad.append(actions_in[bt,:N])

            actions_in = torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            activities_in = activities_in.reshape((batch_size, num_frames))
            activities_in = activities_in[:,0].reshape((batch_size,))
            bboxes_num_in = bboxes_num.reshape(batch_size, num_frames)

            # forward
            actions_scores, activities_scores = model((images_in, boxes_in, bboxes_num_in))
            # activities loss
            activities_loss = F.cross_entropy(activities_scores, activities_in)
            # actions loss
            actions_loss = F.cross_entropy(actions_scores, actions_in, weight=None)  

            # predict labels
            activities_labels = torch.argmax(activities_scores, dim=1)
            actions_labels = torch.argmax(actions_scores, dim=1)  

            # count correct activities
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())
            # count correct actions
            actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())

            # Get accuracies
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]
            actions_accuracy = actions_correct.item() / actions_scores.shape[0]

            # Total loss
            total_loss = activities_loss + actions_loss

            # update meters
            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])
            loss_meter.update(total_loss.item(), batch_size)
            activities_conf.add(activities_labels, activities_in)
            
    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'acties+acts': acties_acts_meter.avg,
        'activities_acc': activities_meter.avg*100,
        'actions_acc': actions_meter.avg*100,
        'activities_acc_side': activities_meter_side.avg*100,
        'activities_acc_team': activities_meter_team.avg*100,
        'activities_conf': activities_conf.value()
    }
    
    return test_info

