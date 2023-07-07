from module.blstm import BidirectionalLSTM
from module.vit import VisionTransformer
import torch.nn as nn

class Tasker(nn.Module):
    def __init__(self, dim):
        super(Tasker, self).__init__()
        self.dim = dim
        self.fc = nn.Linear(self.dim, 100)
    
    def get_feature(self, x):
        pass

    def forward(self, x):
        feat_t = self.get_feature(x)
        return x, feat_t

class img0_tasker(Tasker):
    def __init__(self, dim=1600):
        super(img0_tasker, self).__init__(dim)

    def get_feature(self, x):
        return x[0] # (1600)

class mean_tasker(Tasker):
    def __init__(self, dim=1600):
        super(mean_tasker, self).__init__(dim)

    def get_feature(self, x):
        return x.mean(dim = 0) # (1600)

class blstm_tasker(Tasker):
    def __init__(self, dim=256):
        super(blstm_tasker, self).__init__(dim)
        self.lstm = BidirectionalLSTM(layer_sizes=[self.dim], batch_size=1, vector_dim = 1600)

    def get_feature(self, x):
        #(5, 1, 512) (2, 1, 256) (2, 1, 256)
        output, hn, cn = self.lstm(x.unsqueeze(1)) # (5, 1, 1600) ——> (2, 256)
        feat_t = hn.mean(dim = 0) # (1, 256)
        return feat_t

class vit_tasker(Tasker):
    def __init__(self, dim=1600):
        super(vit_tasker, self).__init__(dim)
        self.vit = VisionTransformer(
                              embed_dim=1600,
                              depth=2,
                              num_heads=16,
                              ).cuda()

    def forward(self, x):
        vit_output = self.vit(x.unsqueeze(0)).squeeze(0)  # (5, 1600)--(1, 5, 1600)--(1, 6, 1600)--(6, 1600)
        cls_token = vit_output[0] # (1600)
        feat_s = vit_output[1:] # (5, 1600)
        return feat_s, cls_token