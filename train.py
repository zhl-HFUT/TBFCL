# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from module.convnet import Convnet
from moco import MoCo
import utils.config as config
from utils.utils import (Timer, count_acc, set_env, MiniImageNet, TrainSampler, load_pretrain, ValSampler)
from utils.logger import Logger
from utils.backup import backup_code
from torch.utils.data import DataLoader

def main():
    set_env(1234)

    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='debug')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--sample_method', default=False)

    # 'mean_tasker' ; 'img0_tasker' ; 'vit_tasker' : 'blstm_tasker'
    parser.add_argument('--tasker', type=str, default='blstm_tasker')

    parser.add_argument('--use_loss_meta', default=True)
    parser.add_argument('--use_loss_taskclassifier', action='store_true')
    parser.add_argument('--use_loss_globalclassifier', action='store_true')
    parser.add_argument('--use_loss_infoNCE', action='store_true')
    parser.add_argument('--use_loss_infoNCE_neg', action='store_true')
    parser.add_argument('--use_loss_sup_con', action='store_true')
    parser.add_argument('--use_loss_blstm_meta', action='store_true')

    parser.add_argument('--use_mlp', action='store_true')

    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--aug_mix', action='store_true')

    parser.add_argument('--memory_size', type=int, default=256)
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
    args.sample_method = args.use_loss_taskclassifier

    backup_code(os.path.dirname(os.path.abspath(__file__)), os.path.join(config.save_path, args.id, 'code'))
    os.makedirs(os.path.join(config.save_path, args.id, 'ckpt'), exist_ok = True)
    logger = Logger(os.path.join(config.save_path, args.id, 'log.txt'))
    logger.log_args(args)

    valset = MiniImageNet('val', config.data_path, twice_sample=False, data_aug=False)

    val_sampler = ValSampler(valset.label, 2000, args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=config.num_workers, pin_memory=True)

    trainset = MiniImageNet('train', config.data_path, twice_sample=True, data_aug=False)

    train_sampler = TrainSampler(trainset.label, 200, args.train_way, args.shot + args.query, args.num_task, args.sample_method)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=config.num_workers, pin_memory=True)

    model = MoCo(Convnet, args.tasker, args.memory_size, args.param_momentum, args.temperature, args.use_mlp).cuda()

    if args.pretrain:
        load_pretrain(model, config.pretrain_conv4)
        logger.log('loaded pretrain model')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):

        train(args, epoch, train_loader, model, optimizer, lr_scheduler, logger)
        if epoch == 1 or epoch % 20 == 0:
            val(args, epoch, val_loader, model, logger)
        model_path = os.path.join(config.save_path, args.id, 'ckpt', f'{epoch}.pth')
        torch.save(model, model_path)
        logger.log('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))

    # best_acc = 0
    # best_epoch = 0
    # for epoch in range(args.max_epoch, 0, -1):
    #     model_path = os.path.join(config.save_path, args.id, 'ckpt', f'{epoch}.pth')
    #     model = torch.load(model_path)
    #     acc = val(args, epoch, val_loader, model, logger)
    #     if acc > best_acc or epoch == 200:
    #         best_acc = acc
    #         best_epoch = epoch
    #         torch.save(model, f'best_model_epoch{epoch}')
    #     logger.log(f'best_epoch:{best_epoch}, acc={best_acc:.4f}')
    #     os.remove(model_path)

def train(args, epoch, train_loader, model, optimizer, lr_scheduler, logger):
    model.train()

    tasks = []
    accs = []
    task_cls_acc = 0
    for i, task in enumerate(train_loader, 1):
        tasks.append(task)
        if i % args.num_task == 0:
            
            loss = 0
            loss_meta = 0
            loss_taskclassifier = 0
            # loss_globalclassifier = 0
            loss_blstm_meta = 0
            loss_infoNCE = 0
            loss_infoNCE_neg = 0
            loss_sup_con = 0
            
            for t in tasks:
                images, labels_all, path, method = [_ for _ in t]
                label = np.array(labels_all)[:5]
                label_category = labels_all[:5].cuda()
                images[0] = images[0].cuda()
                images[1] = images[1].cuda()
                p = args.shot * args.train_way
                s1, q1 = images[0][:p], images[0][p:]
                s2 = images[1][:p]

                logits_meta, labels_meta, logits_taskclassifier, metrics, sims, pure_index, logits_blstm_meta = model(s1, q1, s2, label)

                loss_meta = loss_meta + F.cross_entropy(logits_meta, labels_meta)
                # loss_globalclassifier = loss_globalclassifier + F.cross_entropy(logits_globalclassifier, label_category)
                loss_taskclassifier = loss_taskclassifier + F.cross_entropy(logits_taskclassifier, method[0].cuda())

                loss_blstm_meta = loss_blstm_meta + F.cross_entropy(logits_blstm_meta, labels_meta)

                acc = count_acc(logits_meta, labels_meta)
                accs.append(acc)
                top1_predicted = torch.topk(logits_taskclassifier, 1).indices
                is_correct = method[0].item() in top1_predicted.tolist()
                task_cls_acc += is_correct

                sims = torch.tensor(sims).cuda()
                pure_index = torch.tensor(pure_index).cuda()
                pos_index = []
                for j in range(len(sims)):
                    if sims[j]:
                        pos_index.append(j)
                pos_index = torch.tensor(pos_index).cuda()
                weight_sum = sims.sum()
                metric_exp_sum = torch.exp(metrics).sum()
                label_moco = torch.tensor(0).type(torch.cuda.LongTensor)

                loss_infoNCE = loss_infoNCE + F.cross_entropy(metrics, label_moco)
                loss_infoNCE_neg = loss_infoNCE_neg + F.cross_entropy(torch.index_select(metrics, 0, pure_index), label_moco)
                loss_sup_con = loss_sup_con \
                - (torch.log(torch.exp(torch.index_select(metrics, 0, pos_index)) / metric_exp_sum) * torch.index_select(sims, 0, pos_index)).sum()/weight_sum

            losses = {
                        'loss_meta': (loss_meta/args.num_task, 1.0),
                        'loss_taskclassifier': (loss_taskclassifier/args.num_task, 1.0),
                        # 'loss_globalclassifier': (loss_globalclassifier/args.num_task, 1.0),
                        'loss_blstm_meta': (loss_blstm_meta/args.num_task, 1.0),
                        'loss_infoNCE': (loss_infoNCE/args.num_task, 1.0),
                        'loss_infoNCE_neg': (loss_infoNCE_neg/args.num_task, 1.0),
                        'loss_sup_con': (loss_sup_con/args.num_task, 1.0)
                    }
            # print(loss_meta, loss_taskclassifier, loss_blstm_meta, loss_infoNCE, loss_infoNCE_neg, loss_sup_con)
            
            loss_terms = []
            for loss_name, (loss_value, loss_weight) in losses.items():
                if getattr(args, 'use_' + loss_name):
                    loss = loss + loss_value * loss_weight
                    loss_terms.append("{}*{}({:.4f})".format(loss_weight, loss_name, loss_value))
            loss_sentence = "loss({:.4f}) = ".format(loss) + " + ".join(loss_terms)

            tasks = []
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model._momentum_update_key_encoder()


            if i % 30 == 0:
                logger.log('epoch {}, train {}/{}, L_meta={:.4f}, L_taskcls={:.4f}, '
                    'infoNCE={:.4f}, infoNCE_neg={:.4f}, L_supcon={:.4f}, L_b_m={:.4f}' \
                    .format(epoch, i, len(train_loader) * args.num_task, \
                    loss_meta.item()/args.num_task, loss_taskclassifier.item()/args.num_task, \
                    loss_infoNCE.item()/args.num_task, loss_infoNCE_neg.item()/args.num_task, loss_sup_con.item()/args.num_task, loss_blstm_meta.item()/args.num_task))

                logger.log(loss_sentence)

                # verify task feature classify
                logger.log('Sample method predicted: {}, GT: {}, {}' \
                    .format(', '.join(map(str, top1_predicted.tolist())), method[0].item(), "True" if is_correct else "False"))

                # verify global classify
                # pred = torch.argmax(logits_globalclassifier, dim=1)
                # print('Global classifier predicted: [{}],' \
                #     .format(', '.join(map(str, pred.tolist()))), 'GT: [{}],'.format(', '.join(map(str, label))), \
                #     'acc={:.4f}'.format(count_acc(logits_globalclassifier, label_category)))

    mean = np.mean(accs) * 100
    std = (1.96 * np.std(accs, ddof=1) / np.sqrt(len(train_loader))) * 100

    logger.log('Train set few-shot acc={:.4f}±{:.4f}'.format(mean, std))
    logger.log(f'Task classifier acc={task_cls_acc}/{len(train_loader) * args.num_task}')
    lr_scheduler.step()


def val(args, epoch, val_loader, model, logger):
    model.eval()

    # val_loss = Averager()
    # val_acc = Averager()

    accs = []

    for i, task in enumerate(val_loader, 1):
        images, labels_all, path, method = [_ for _ in task]
        label_category = np.array(labels_all)[:5]
        images = images.cuda()
        p = args.shot * args.test_way
        s1, q1 = images[:p], images[p:]

        logits_meta, labels_meta, logits_blstm_meta = model(s1, q1, s1, label_category)

        # loss_meta = F.cross_entropy(logits_meta, labels_meta)
        accs.append(count_acc(logits_meta, labels_meta))

        # val_loss.add(loss_meta.item())
        # val_acc.add(acc)

    # loss_meta = val_loss.item()
    # std = (1.96 * np.std(accs, ddof=1) / np.sqrt(len(val_loader))) * 100

    logger.log('epoch {}, acc={:.4f}'.format(epoch, np.mean(accs) * 100))


if __name__ == '__main__':
    main()