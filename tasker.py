from module.blstm import BidirectionalLSTM
from module.vit import VisionTransformer
import torch.nn as nn
import torch

class Tasker(nn.Module):
    def __init__(self, dim, use_mlp):
        super(Tasker, self).__init__()
        self.dim = dim
        self.use_mlp = use_mlp
        self.fc = nn.Linear(self.dim, 100)
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim)
            )
    
    def get_feature(self, x):
        pass

    def forward(self, x):
        feat_t = self.get_feature(x)
        if self.use_mlp:
            feat_t = self.mlp(feat_t)
        return x, feat_t

class img0_tasker(Tasker):
    def __init__(self, use_mlp, dim=1600, ):
        super(img0_tasker, self).__init__(dim, use_mlp)

    def get_feature(self, x):
        return x[0] # (1600)

class mean_tasker(Tasker):
    def __init__(self, use_mlp, dim=1600):
        super(mean_tasker, self).__init__(dim, use_mlp)

    def get_feature(self, x):
        return x.mean(dim = 0) # (1600)

class blstm_tasker(Tasker):
    def __init__(self, use_mlp, dim=256):
        super(blstm_tasker, self).__init__(dim, use_mlp)
        self.lstm = BidirectionalLSTM(layer_sizes=[dim], batch_size=1, vector_dim = 1600)

    def get_feature(self, x):
        #(5, 1, 512) (2, 1, 256) (2, 1, 256)
        output, hn, cn = self.lstm(x.unsqueeze(1)) # (5, 1, 1600) ——> (2, 256)
        # print(output.shape, hn.shape, cn.shape)
        feat_t = hn.mean(dim = 0) # (1, 256)

        # maxpooling
        # feat_t, _ =  torch.max(output, dim=0) # (1, 512)
        # print(feat_t.shape)
        return output, feat_t
    
    def forward(self, x):
        output, feat_t = self.get_feature(x)
        if self.use_mlp:
            feat_t = self.mlp(feat_t)
        return feat_t, output.squeeze(1)

class vit_tasker(Tasker):
    def __init__(self, use_mlp, dim=1600):
        super(vit_tasker, self).__init__(dim, use_mlp)
        self.vit = VisionTransformer(
                              embed_dim=1600,
                              depth=2,
                              num_heads=4,
                              ).cuda()

    def forward(self, x):
        vit_output = self.vit(x.unsqueeze(0)).squeeze(0)  # (5, 1600)--(1, 5, 1600)--(1, 6, 1600)--(6, 1600)
        cls_token = vit_output[0].unsqueeze(0) # (1, 1600)
        # print(cls_token.shape)
        feat_s = vit_output[1:] # (5, 1600)
        return cls_token, feat_s