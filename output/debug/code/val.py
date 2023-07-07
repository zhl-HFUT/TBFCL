import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from module.convnet import Convnet
from moco import MoCo
import utils.config as config
from utils.utils import (Averager, Timer, count_acc, set_env, MiniImageNet, TrainSampler, load_pretrain, ValSampler)
from utils.logger import Logger
from utils.backup import backup_code
from torch.utils.data import DataLoader

def main():
    set_env(1234)

    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='2-3')
    parser.add_argument('--pretrain', type=bool, default=True)

    # 'mean_tasker' ; 'img0_tasker' ; 'vit_tasker' : 'blstm_tasker'
    parser.add_argument('--tasker', type=str, default='blstm_tasker')

    parser.add_argument('--memory_size', type=int, default=128)
    parser.add_argument('--param_momentum', type=float, default=0.99)
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

    val_sampler = ValSampler(valset.label, 2000, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=config.num_workers, pin_memory=True)

    model_path = '/root/code/TBFCL/output/2-5-test/ckpt/195.pth'
    print(model_path)
    model = torch.load(model_path)
    acc = val(args, val_loader, model)
    print(acc)


def val(args, val_loader, model):
    model.eval()

    val_loss = Averager()
    val_acc = Averager()

    accs = []
    for i, task in enumerate(val_loader, 1):
        images, labels_all, path, method = [_ for _ in task]
        label_category = np.array(labels_all)[:5]
        images = images.cuda()
        p = args.shot * args.test_way
        s1, q1 = images[:p], images[p:]

        logits_meta, labels_meta = model(s1, q1, s1, label_category)
        loss_meta = F.cross_entropy(logits_meta, labels_meta)
        acc = count_acc(logits_meta, labels_meta)

        val_loss.add(loss_meta.item())
        val_acc.add(acc)
        accs.append(acc)

    loss_meta = val_loss.item()
    mean = np.mean(accs) * 100
    std = (1.96 * np.std(accs, ddof=1) / np.sqrt(len(val_loader))) * 100

    print('loss-images={:.4f} acc={:.4f}±{:.4f}' \
        .format(loss_meta, mean, std))
    
    return mean


if __name__ == '__main__':
    main()