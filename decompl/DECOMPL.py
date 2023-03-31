from backbone.backbone import *
from utils import *
from torchvision.ops import ps_roi_align
from typing import Tuple, Optional, Callable, List, TYPE_CHECKING
from config import *


"""
Reference:
https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark/blob/4648310a42ca7b66013da9d623e9f856a483f30c/base_model.py
"""

class Spatial_Block(nn.Module):
    def __init__(self):
        super(Spatial_Block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4, stride=4) # 1 16
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding="same") # 16 16
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding="same") # 16 16

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding="same") # 16 16
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding="same") # 16 16
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding="same") # 16 16

        self.fc = nn.Linear(in_features=384, out_features=128) # 160 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        x = x.unsqueeze(dim=1)
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        pool_out = self.pool(conv2_out)
        conv3_out = self.conv3(pool_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        pool_out2 = self.pool(conv5_out)
        conv6_out = self.conv6(pool_out2).reshape(B,-1)
        fc_out = self.fc(conv6_out)
        return fc_out


class Attention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int=512):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(in_features = self.L, out_features = self.D),
            nn.Tanh(),
            nn.Linear(in_features = self.D, out_features = 1)
        )

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = self.attention(H)  # batch_size x K x 1
        A = torch.transpose(A, 2, 3)  # batch_size x 1 x K
        A = F.softmax(A, dim=2)  # softmax over dim 2
        M = torch.matmul(A, H)  # batch_size x 1 x K * K x m -> batch_size x m
        return M.squeeze(), A.squeeze()


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.msa = nn.ModuleList([Attention(input_dim) for _ in range(num_heads)])
        self.fc = nn.Linear(input_dim*num_heads, input_dim)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B,T,N,D = H.shape
        out = []
        attn_weights = []
        for layer in self.msa:
            M, A = layer(H)
            out.append(M.squeeze().view(B,T,D))
            attn_weights.append(A.squeeze())
        out = torch.cat(out, dim=2)
        attn_weights = torch.stack(attn_weights)
        out = self.fc(out)
        return out # , attn_weights


class DECOMPL_volleyball(nn.Module):
    def __init__(self, cfg: Config):
        super(DECOMPL_volleyball, self).__init__()
        self.cfg=cfg
        NFB = self.cfg.num_features_boxes
        D = self.cfg.emb_features
        self.K = self.cfg.crop_size[0] # for RoI Align

        #################################################################################
        ############################# VISUAL BRANCH #####################################
        #################################################################################
        # Backbone
        self.backbone = MyVGG16(pretrained=True)
        # Embedding Layer
        self.fc_emb = nn.Linear(D, NFB)
        self.dropout_emb = nn.Dropout(p=self.cfg.train_dropout_prob)
        
        # Multi-Head Self Attention Pooling
        self.num_heads = 2
        self.left_attention = MultiHeadAttention(NFB, self.num_heads)
        self.right_attention = MultiHeadAttention(NFB, self.num_heads)

        # Visual Classifiers
        self.fc_actions = nn.Linear(NFB,self.cfg.num_actions)
        self.fc_activities = nn.Linear(2*NFB,self.cfg.num_activities)
        self.fc_activities_side = nn.Linear(2*NFB,2)
        self.fc_activities_team = nn.Linear(2*NFB,4)

        #################################################################################
        ############################# SPATIAL BRANCH ####################################
        #################################################################################
        # Spatial Block
        self.spatial_block = Spatial_Block()
        
        # Single Head Attention Pooling
        self.shared_attention = Attention(NFB)

        # Batchnorm to scale different branch's outputs
        self.batchnorm = nn.BatchNorm1d(num_features = cfg.num_boxes)

        # Spatial Classifiers
        self.fc_activities_side_spat = nn.Sequential(nn.Linear(NFB, 64), nn.ReLU(), nn.Linear(64, 2))
        self.fc_activities_team_spat = nn.Sequential(nn.Linear(NFB, 64), nn.ReLU(), nn.Linear(64, 4))
        self.fc_activities_spat = nn.Sequential(nn.Linear(NFB, 64), nn.ReLU(), nn.Linear(64, 8))

        # Fusion     
        self.side_fusion = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.team_fusion = nn.Parameter(torch.zeros(4), requires_grad=True)
        self.activity_fusion = nn.Parameter(torch.zeros(8), requires_grad=True)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images_in, boxes_in = batch_data
        # read config parameters
        B = images_in.shape[0] # batch size
        T = images_in.shape[1] # temporal dimension
        H, W = self.cfg.image_size # img dim
        OH, OW = self.cfg.out_size # scaling VGG output
        N = self.cfg.num_boxes # num players
        NFB = self.cfg.num_features_boxes # embedding dim
        
        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B*T, 3, H, W))  #B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B*T*N, 4))  #B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B*T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (B*T*N,))  #B*T*N,
        

        #################################################################################
        ############################# VISUAL BRANCH #####################################
        #################################################################################
        
        images_in_flat = prep_images(images_in_flat) # preprocess image
        outputs = self.backbone(images_in_flat) # VGG output
        
        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features,size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale = torch.cat(features_multiscale, dim=1)  #B*T, D, OH, OW
        boxes_in_flat = torch.cat((boxes_idx_flat.reshape(-1,1), boxes_in_flat), dim=1) 

        boxes_features = ps_roi_align(features_multiscale, boxes_in_flat, (self.K, self.K)) # B*T*N, D/K*K, K,K
        boxes_features = boxes_features.reshape(B*T*N, -1) # B*T*N, D
        
        # Embedding to hidden state
        boxes_features = self.fc_emb(boxes_features)  # B*T*N, NFB

        boxes_features = F.relu(boxes_features)
        boxes_features = self.dropout_emb(boxes_features)
        boxes_states = boxes_features.reshape(B, T, N, NFB)

        # Sort boxes from left to right
        sorted_indices = [torch.argsort(el[:,0]) for el in boxes_in.view(B*T, N, -1)] # argsort wrt x coord in the group
        boxes_states_permuted = torch.stack([el[id] for el,id in zip(boxes_states.view(B*T, N, -1), sorted_indices)]).reshape(B, T, N, NFB)
        first_team_boxes_states = boxes_states_permuted[:, :, :6, :] # extract the first team's feature map
        second_team_boxes_states = boxes_states_permuted[:, :, 6: ,:] # extract the second team's feature map

        # Attention pooling for both teams
        first_team_att = self.left_attention(first_team_boxes_states) # left attention
        second_team_att = self.right_attention(second_team_boxes_states) # right attention
        together_features = torch.cat((first_team_att.view(B*T, -1), second_team_att.view(B*T, -1)), dim=1)

        # Predict activities
        activities_scores = self.fc_activities(together_features)  #B*T, acty_num
        activities_scores_side = self.fc_activities_side(together_features) #B*T, side_acty_num
        activities_scores_team = self.fc_activities_team(together_features)  #B*T, team_acty_num

        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1, NFB)  #B*T*N, NFB
        actions_scores = self.fc_actions(boxes_states_flat)  #B*T*N, actn_num


        #################################################################################
        ############################# SPATIAL BRANCH ####################################
        #################################################################################
        
        # prepare spatial info
        boxes_sorted = torch.stack([el[id] for el, id in zip(boxes_in.view(B*T, N, -1), sorted_indices)]).reshape(B, T, N, -1)
        # extract spatial features
        spatial_differences = torch.stack([torch.stack([b1-b2 for b1 in boxes_sorted_instance.view(N, -1) for b2 in boxes_sorted_instance.view(N, -1)]) for boxes_sorted_instance in boxes_sorted.view(B*T, N, -1)]).reshape(B*T, N*N*4)
        spatial_diff_normed = torch.stack([2*(el-torch.min(el))/(torch.max(el)-torch.min(el))-1 for el in spatial_differences]).reshape(B*T*N, N*4)
        spatial_features = self.spatial_block(spatial_diff_normed)
        spatial_features = self.batchnorm(spatial_features.view(B*T, N, NFB)).reshape(B, T, N, -1)
        together_features_spat, _ = self.shared_attention(spatial_features)

        activities_scores_spat = self.fc_activities_spat(together_features_spat).reshape(B*T, -1)  #B*T, acty_num
        activities_scores_side_spat = self.fc_activities_side_spat(together_features_spat).reshape(B*T, -1) #B*T, side_acty_num
        activities_scores_team_spat = self.fc_activities_team_spat(together_features_spat).reshape(B*T, -1)  #B*T, team_acty_num
        
        # Fusion
        activities_scores_side += torch.multiply(activities_scores_side_spat, self.side_fusion)
        activities_scores_team += torch.multiply(activities_scores_team_spat, self.team_fusion)
        activities_scores += torch.multiply(activities_scores_spat, self.activity_fusion)

        if T != 1: # for inference
            actions_scores = actions_scores.reshape(B, T, N, -1).mean(dim=1).reshape(B*N, -1)
            activities_scores = activities_scores.reshape(B, T, -1).mean(dim=1)
            activities_scores_side = activities_scores_side.reshape(B, T, -1).mean(dim=1)
            activities_scores_team = activities_scores_team.reshape(B, T, -1).mean(dim=1)
            
        return actions_scores, activities_scores_side, activities_scores_team, activities_scores
    


class DECOMPL_collective(nn.Module):
    def __init__(self, cfg: Config):
        super(DECOMPL_collective, self).__init__()
        self.cfg=cfg

        NFB = self.cfg.num_features_boxes
        D = self.cfg.emb_features
        self.K = self.cfg.crop_size[0] # for RoI Align

        #################################################################################
        ############################# VISUAL BRANCH #####################################
        #################################################################################
        # Backbone
        self.backbone = MyVGG16(pretrained=True)
        # Embedding Layer
        self.fc_emb = nn.Linear(D, NFB)
        self.dropout_emb = nn.Dropout(p=self.cfg.train_dropout_prob)
        
        # Multi-Head Self Attention Pooling
        self.num_heads = 2
        self.attention = MultiHeadAttention(NFB, self.num_heads)

        # Visual Classifiers
        self.fc_actions = nn.Linear(NFB, self.cfg.num_actions)
        self.fc_activities = nn.Linear(NFB, self.cfg.num_activities)

        #################################################################################
        ############################# SPATIAL BRANCH ####################################
        #################################################################################
        # Spatial Block
        self.spatial_block = Spatial_Block()
        
        # Single Head Attention Pooling
        self.shared_attention = Attention(NFB)

        # Batchnorm to scale different branch's outputs
        self.batchnorm = nn.BatchNorm1d(num_features = cfg.num_boxes)

        # Spatial Classifiers
        self.fc_activities_side_spat = nn.Sequential(nn.Linear(NFB, 64), nn.ReLU(), nn.Linear(64, 2))
        self.fc_activities_team_spat = nn.Sequential(nn.Linear(NFB, 64), nn.ReLU(), nn.Linear(64, 4))
        self.fc_activities_spat = nn.Sequential(nn.Linear(NFB, 64), nn.ReLU(), nn.Linear(64, 8))

        # Fusion     
        self.side_fusion = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.team_fusion = nn.Parameter(torch.zeros(4), requires_grad=True)
        self.activity_fusion = nn.Parameter(torch.zeros(8), requires_grad=True)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)



    def forward(self, batch_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        images_in, boxes_in, bboxes_num_in = batch_data
        # read config parameters
        B = images_in.shape[0] # batch size
        T = images_in.shape[1] # temporal dimension
        H, W = self.cfg.image_size # img dim
        OH, OW = self.cfg.out_size # scaling VGG output
        MAX_N = self.cfg.num_boxes # num players
        NFB = self.cfg.num_features_boxes # embedding dim
        
        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B*T, 3, H, W))  #B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (B*T*MAX_N, 4))  #B*T*N, 4

        #################################################################################
        ############################# VISUAL BRANCH #####################################
        #################################################################################
        
        images_in_flat = prep_images(images_in_flat) # preprocess image
        outputs = self.backbone(images_in_flat) # VGG output
        
        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features,size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale = torch.cat(features_multiscale, dim=1)  #B*T, D, OH, OW
        
        bboxes_num_in = bboxes_num_in.reshape(B*T)
        boxes_idx = [i * torch.ones(bboxes_num_in[i], dtype=torch.int) for i in range(B*T)]
        boxes_idx = torch.cat(boxes_idx, dim=0).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (-1,))  #B*T*N,

        boxes_in_flat = [boxes[:bboxes_num_in[i], :] for i, boxes in enumerate(boxes_in.view(B*T, MAX_N, -1))]
        boxes_in_flat = torch.cat(boxes_in_flat, dim=0)
        boxes_in_flat = torch.cat((boxes_idx_flat.reshape(-1,1), boxes_in_flat), dim=1) 
        boxes_features_all = ps_roi_align(features_multiscale, boxes_in_flat, (self.K, self.K)) # B*T*N, D/K*K, K,K
        boxes_features_all = boxes_features_all.reshape(-1, 512) # B*T*N, D
        # Embedding to hidden state
        boxes_features_all = self.fc_emb(boxes_features_all)  # B*T*N, NFB

        boxes_features_all = F.relu(boxes_features_all)
        boxes_features_all = self.dropout_emb(boxes_features_all)

        actions_scores = []
        activities_scores = []
        bboxes_num_in = bboxes_num_in.reshape(B*T,)  #B*T,
        count = 0
        for b in range(B):
            actn_scores = []
            for t in range(T):
                bt = b*T + t
                N = bboxes_num_in[bt]
                boxes_features = boxes_features_all[count:count+N,:].reshape(1,1,N,NFB)  #1,N,NFB
                count += N
                boxes_states = boxes_features  
                NFS=NFB
                # Predict actions
                boxes_states_flat = boxes_states.reshape(-1,NFS)  #1*N, NFS
                actn_score = self.fc_actions(boxes_states_flat)  #1*N, actn_num
                actn_scores.append(actn_score)
                # Predict activities
                boxes_states_pooled = self.attention(boxes_states)
                boxes_states_pooled_flat=boxes_states_pooled.reshape(-1,NFS)  #1, NFS
                acty_score = self.fc_activities(boxes_states_pooled_flat)  #1, acty_num
                activities_scores.append(acty_score)
            actn_scores = torch.stack(actn_scores)
            actions_scores.append(torch.mean(actn_scores, dim=0))
        
        actions_scores = torch.cat(actions_scores,dim=0)  #ALL_N,actn_num
        activities_scores = torch.cat(activities_scores,dim=0)   #B*T,acty_num

        if T != 1: # for inference
            activities_scores = activities_scores.reshape(B, T, -1).mean(dim=1)
            
        return actions_scores, activities_scores