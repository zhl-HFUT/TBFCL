import numpy as np
import torch
import torch.nn as nn
from module.blstm import BidirectionalLSTM
from module.vit import VisionTransformer
from tasker import mean_tasker, img0_tasker, vit_tasker, blstm_tasker

class MoCo(nn.Module):
    def __init__(self, base_encoder, tasker, memory_size, param_momentum, temperature, use_mlp):
        super(MoCo, self).__init__()

        self.K = memory_size
        self.m = param_momentum
        self.T = temperature
        
        self.encoder = base_encoder()
        self.tasker_A = eval(tasker)(use_mlp)
        
        self.encoder_B = base_encoder()
        self.tasker_B = eval(tasker)(use_mlp)

        # self.fc = nn.Linear(1600, 64)

        for param, param_k in zip(self.encoder.parameters(), self.encoder_B.parameters()):
            param_k.data.copy_(param.data)
            param_k.requires_grad = False
        
        for param, param_k in zip(self.tasker_A.parameters(), self.tasker_B.parameters()):
            param_k.data.copy_(param.data)
            param_k.requires_grad = False
        
        # create the queue
        self.register_buffer("queue", torch.randn(self.K, self.tasker_A.dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # classes of task in quene
        self.classes = np.ones((self.K, 5), dtype=int)*1000
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # print('update key encoder')
        for param, param_k in zip(self.encoder.parameters(), self.encoder_B.parameters()):
            param_k.data = param_k.data * self.m + param.data * (1. - self.m)
        if isinstance(self.tasker_A, blstm_tasker):
            for param, param_k in zip(self.tasker_A.parameters(), self.tasker_B.parameters()):
                param_k.data = param_k.data * self.m + param.data * (1. - self.m)
        if isinstance(self.tasker_A, vit_tasker):
            for param, param_k in zip(self.tasker_A.parameters(), self.tasker_B.parameters()):
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

    def forward(self, support_A, query_A, support_B, key_cls):
        # # shuffle support set for 
        # indices = torch.randperm(s1.size(0)).cuda()
        # shuffled_s1 = s1[indices]
        # print(indices)

        # base
        feat_support_A = self.encoder(support_A) # (5, 1600)
        # logits_globalclassifier = self.fc(feat_support_A)

        # anchor feat_task
        feat_task_A, blstm_output_A = self.tasker_A(feat_support_A)
        feat_task_A = nn.functional.normalize(feat_task_A, dim=1) # (1, 256)

        # for meta learning
        feat_support_A = nn.functional.normalize(feat_support_A, dim=1)
        feat_query_A = self.encoder(query_A) # (75, 1600)
        labels_meta = torch.tensor(np.arange(feat_support_A.shape[0])).repeat(int(feat_query_A.shape[0]/feat_support_A.shape[0])).type(torch.cuda.LongTensor)
        logits_meta = torch.mm(feat_query_A, feat_support_A.t())

        # for extra blstm output
        tensor_list = []
        for feat in feat_query_A:
            _, blstm_query = self.tasker_A(feat.unsqueeze(0))
            tensor_list.append(blstm_query)
        stacked_tensor = torch.stack(tensor_list)
        reshaped_tensor = stacked_tensor.view(75, 512)
        logits_blstm_meta = torch.mm(reshaped_tensor, blstm_output_A.t())
        # accs4.append(count_acc(logits_blstm, labels_meta))
        # _, blstm_output_q = self.tasker(feat_q)
        # logits_meta_lstm = torch.mm(blstm_output_q, blstm_output_s.t())

        # for contrastive
        if self.training:
            # for task classify
            logits_taskclassifier = self.tasker_A.fc(feat_task_A.squeeze(0))
            with torch.no_grad():
                # positive feat_task
                feat_support_B = self.encoder_B(support_B) # (5, 1600)
                feat_task_B, _ = self.tasker_B(feat_support_B)
                feat_task_B = nn.functional.normalize(feat_task_B, dim=1) # (1, 256)

            metric_pos = torch.dot(feat_task_A.squeeze(0), feat_task_B.squeeze(0)).unsqueeze(-1)
            metric_memory = torch.mm(feat_task_A, self.queue.clone().detach().t())
            metrics = torch.cat((metric_pos, metric_memory.squeeze(0)), dim=0)
            metrics /= self.T

            sims = [1]
            pure_index = [0]
            for i in range(self.K):
                sims.append(len(np.intersect1d(self.classes[i,:], key_cls))/5.)
                if not bool(len(np.intersect1d(self.classes[i,:], key_cls))):
                    pure_index.append(i+1)
            self._dequeue_and_enqueue(feat_task_B, key_cls)
            return logits_meta, labels_meta, logits_taskclassifier, metrics, sims, pure_index, logits_blstm_meta

        else:
            # feat_s, feat_t = self.tasker(feat_s)
            # feat_s_external = self.model_external(support_A)
            # feat_s_external = nn.functional.normalize(feat_s_external, dim=1)
            # feat_q_external = self.model_external(query_A)
            # logits_external = torch.mm(feat_q_external, feat_s_external.t())
            return logits_meta, labels_meta, logits_blstm_meta