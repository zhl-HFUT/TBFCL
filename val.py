import argparse

import numpy as np
import torch
import torch.nn.functional as F
from module.convnet import Convnet
from moco import MoCo
import utils.config as config
from utils.utils import (Averager, Timer, count_acc, set_env, MiniImageNet, TrainSampler, load_pretrain, ValSampler)
from torch.utils.data import DataLoader

import random
from PIL import ImageFilter
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

import torchvision.transforms as transforms
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]


def main():
    set_env(1234)

    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='debug')
    parser.add_argument('--pretrain', action='store_true')

    # 'mean_tasker' ; 'img0_tasker' ; 'vit_tasker' : 'blstm_tasker'
    parser.add_argument('--tasker', type=str, default='blstm_tasker')

    parser.add_argument('--use_loss_meta', default=True)
    parser.add_argument('--use_loss_taskclassifier', action='store_true')
    parser.add_argument('--use_loss_globalclassifier', action='store_true')
    parser.add_argument('--use_loss_infoNCE', action='store_true')
    parser.add_argument('--use_loss_infoNCE_neg', action='store_true')
    parser.add_argument('--use_loss_sup_con', action='store_true')

    parser.add_argument('--use_mlp', default=True)

    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--aug_mix', action='store_true')

    parser.add_argument('--memory_size', type=int, default=256)
    parser.add_argument('--param_momentum', type=float, default=0.999)
    parser.add_argument('--temperature', type=float, default=0.07)

    parser.add_argument('--max_epoch', type=int, default=200)
    # 1 ; 5(to do)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train_way', type=int, default=5)
    parser.add_argument('--test_way', type=int, default=5)
    
    # 6 ; 8(to do)
    parser.add_argument('--num_task', type=int, default=6)

    args = parser.parse_args()

    valset = MiniImageNet('val', config.data_path, twice_sample=False)

    val_sampler = ValSampler(valset.label, 100, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=config.num_workers, pin_memory=True)

    trainset = MiniImageNet('train', config.data_path, twice_sample=True)

    # model = MoCo(Convnet, args.tasker, args.memory_size, args.param_momentum, args.temperature, args.use_mlp).cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # timer = Timer()
    # model_external = torch.load('/root/code/TBFCL/checkpoints/41_132.pth')
    model = torch.load('/root/code/TBFCL/checkpoints/79_75.pth')
    # model2 = torch.load('/root/code/TBFCL/output/local-1/ckpt/200.pth')
    # print(model.tasker.use_mlp)
    # model.tasker.use_mlp = True
    # model.eval()
    conv_q = model.encoder
    conv_k = model.encoder_k
    val(args, val_loader, model, conv_q, conv_k)

import torch.nn as nn

def val(args, val_loader, model, conv_q, conv_k):
    random.seed(42)

    accs = []
    accs_k = []
    accs_qk = []
    # accs_blstm = []
    # accs_mix = []
    # accs_80 = []
    # accs_mix80 = []
    for i, task in enumerate(val_loader, 1):
        images, images_aug, labels_all, path, method = [_ for _ in task]
        label_category = np.array(labels_all)[:5]
        images = images.cuda()
        p = args.shot * args.test_way
        s1, q1 = images[:p], images[p:]

        # logits_meta, labels_meta, logits_meta_lstm, logits_external, logits_meta_k = model(s1, q1, s1, label_category)
        # print(logits_meta.shape, labels_meta.shape, logits_meta_lstm.shape, logits_external.shape)

        # # 创建随机索引
        # indices = list(range(len(logits_meta)))
        # random.shuffle(indices)

        # # 使用随机索引对所有张量进行重新排序
        # logits_meta_shuffled = logits_meta[indices]
        # labels_shuffled = labels_meta[indices]
        # logits_meta_lstm_shuffled = logits_meta_lstm[indices]
        # logits_external_shuffled = logits_external[indices]

        # acc = count_acc(logits_meta, labels_meta)

        feat_s = conv_q(s1) # (5, 1600)
        feat_s = nn.functional.normalize(feat_s, dim=1)
        feat_s_k = conv_k(s1) # (5, 1600)
        feat_s_k = nn.functional.normalize(feat_s_k, dim=1)

        feat_q = conv_q(q1)
        feat_q_k = conv_k(q1)

        logits_meta = torch.mm(feat_q, feat_s.t())
        logits_meta_k = torch.mm(feat_q_k, feat_s_k.t())

        labels_meta = torch.tensor(np.arange(feat_s.shape[0])).repeat(int(feat_q.shape[0]/feat_s.shape[0])).type(torch.cuda.LongTensor)

        accs.append(count_acc(logits_meta, labels_meta))
        accs_k.append(count_acc(logits_meta_k, labels_meta))
        accs_qk.append(count_acc(logits_meta+logits_meta_k, labels_meta))
        # accs_blstm.append(count_acc(logits_meta_lstm, labels_meta))
        # accs_mix.append(count_acc(logits_meta+logits_meta_lstm, labels_meta))
        # accs_80.append(count_acc(logits_external, labels_meta))
        # accs_mix80.append(count_acc(logits_meta+logits_external, labels_meta))
        # accs.append(count_acc(logits_meta_shuffled, labels_shuffled))
        # accs_blstm.append(count_acc(logits_meta_lstm_shuffled, labels_shuffled))
        # accs_mix.append(count_acc(logits_meta_shuffled+logits_meta_lstm_shuffled, labels_shuffled))
        # accs_80.append(count_acc(logits_external_shuffled, labels_shuffled))
        # accs_mix80.append(count_acc(logits_meta_shuffled+logits_external_shuffled, labels_shuffled))


    print('acc={:.4f}'.format(np.mean(accs) * 100))
    print('acc_k={:.4f}'.format(np.mean(accs_k) * 100))
    print('acc_qk={:.4f}'.format(np.mean(accs_qk) * 100))
    # print('acc_blstm={:.4f}'.format(np.mean(accs_blstm) * 100))
    # print('acc_mix={:.4f}'.format(np.mean(accs_mix) * 100))
    # print('acc_80={:.4f}'.format(np.mean(accs_80) * 100))
    # print('acc_mix80={:.4f}'.format(np.mean(accs_mix80) * 100))


if __name__ == '__main__':
    main()