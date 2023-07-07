import numpy as np
import torch
import torch.nn as nn
from module.blstm import BidirectionalLSTM
from module.vit import VisionTransformer
from tasker import mean_tasker, img0_tasker, vit_tasker, blstm_tasker

class MoCo(nn.Module):
    def __init__(self, base_encoder, tasker, memory_size, param_momentum, temperature):
        super(MoCo, self).__init__()

        self.K = memory_size
        self.m = param_momentum
        self.T = temperature
        
        self.encoder = base_encoder()
        self.tasker = eval(tasker)()
        self.encoder_k = base_encoder()
        self.tasker_k = eval(tasker)()

        self.fc = nn.Linear(1600, 64)

        for param, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param.data)
            param_k.requires_grad = False
        
        for param, param_k in zip(self.tasker.parameters(), self.tasker_k.parameters()):
            param_k.data.copy_(param.data)
            param_k.requires_grad = False
        
        # create the queue
        self.register_buffer("queue", torch.randn(self.K, self.tasker.dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # classes of task in quene
        self.classes = np.ones((self.K, 5), dtype=int)*1000
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param.data * (1. - self.m)
        if isinstance(self.tasker, blstm_tasker):
            for param, param_k in zip(self.tasker.parameters(), self.tasker_k.parameters()):
                param_k.data = param_k.data * self.m + param.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key, key_cls):
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr] = key
        self.classes[ptr] = key_cls
        # move pointer
        ptr = (ptr + 1) % self.K  
        self.queue_ptr[0] = ptr

    def forward(self, s1, q1, s2, key_cls):
        # anchor feat_t
        feat_s = self.encoder(s1) # (5, 1600)
        feat_s, feat_t = self.tasker(feat_s)
        feat_t = nn.functional.normalize(feat_t, dim=1) # (1, 256)
        # print('test sims')

        # for meta learning
        feat_s = nn.functional.normalize(feat_s, dim=1)
        feat_q = self.encoder(q1) # (75, 1600)

        logits_meta = torch.mm(feat_q, feat_s.t())
        labels_meta = torch.tensor(np.arange(feat_s.shape[0])).repeat(int(feat_q.shape[0]/feat_s.shape[0])).type(torch.cuda.LongTensor)

        # for contrastive
        if self.training:
            # for task classify
            logits_taskclassifier = self.tasker.fc(feat_t.squeeze(0))
            logits_globalclassifier = self.fc(feat_s)
            with torch.no_grad():
                # positive feat_t
                feat_s_pos = self.encoder(s2) # (5, 1600)
                feat_s_pos, feat_t_pos = self.tasker_k(feat_s_pos)
                feat_t_pos = nn.functional.normalize(feat_t_pos, dim=1) # (1, 256)

            metric_pos = torch.dot(feat_t.squeeze(0), feat_t_pos.squeeze(0)).unsqueeze(-1)
            metric_memory = torch.mm(feat_t, self.queue.clone().detach().t())
            metrics = torch.cat((metric_pos, metric_memory.squeeze(0)), dim=0)
            metrics /= self.T

            sims = [1]
            pure_index = [0]
            for i in range(self.K):
                sims.append(len(np.intersect1d(self.classes[i,:], key_cls))/5.)
                if not bool(len(np.intersect1d(self.classes[i,:], key_cls))):
                    pure_index.append(i+1)
            self._dequeue_and_enqueue(feat_t_pos, key_cls)
            return logits_meta, labels_meta, logits_globalclassifier, logits_taskclassifier, metrics, sims, pure_index

        else:
            return logits_meta, labels_meta

