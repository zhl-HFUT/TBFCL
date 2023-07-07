import os
import os.path as osp
from PIL import Image
import time

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import random
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

class TrainSampler():

    def __init__(self, label, n_batch, n_cls, n_per, n_task):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.n_task = n_task

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            for i in range(self.n_task):
                methods = []
                samples = []
                # classes = torch.randperm(len(self.m_ind))[:self.n_cls]
                method = random.randint(0, 99)
                classes = torch.tensor(config.combinations[method])
                for c in classes:
                    l = self.m_ind[c]
                    pos = torch.randperm(len(l))[:self.n_per]
                    samples.append(l[pos])
                    methods.extend([method] * self.n_per)
                samples = torch.stack(samples).t().reshape(-1)
                methods = torch.tensor(methods)
                task = list(zip(samples, methods))

                yield task

class ValSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        self.all_tasks = []
        for i_batch in range(self.n_batch):
            task = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                task.append(l[pos])
            task = torch.stack(task).t().reshape(-1)
            self.all_tasks.append(task)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            yield self.all_tasks[i_batch]


class MiniImageNet(Dataset):

    def __init__(self, setname, data_path, twice_sample):
        csv_path = osp.join(data_path, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(data_path, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.data_path = data_path

        self.transform = transforms.Compose([
                transforms.Resize(84),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        self.twice_sample = twice_sample
        self.lines = [x.strip().split(',') for x in open(csv_path, 'r').readlines()][1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, keys):
        try:
            i, method = keys
        except:
            i = keys
            method = -1
        path, label = self.data[i], self.label[i]

        if self.twice_sample == True:
            index = random.randint(int(label)*600, (int(label)+1)*600 - 1) 
            new_path = osp.join(self.data_path, 'images', self.lines[index][0])

            while path == new_path:
                index = random.randint(int(label)*600, (int(label)+1)*600 - 1)
                new_path = osp.join(self.data_path, 'images', self.lines[index][0])

            with open(path, "rb") as f:
                img1 = Image.open(f)
                img1 = img1.convert("RGB")
            with open(new_path, "rb") as f:
                img2 = Image.open(f)
                img2 = img2.convert("RGB")
            image1 = self.transform(img1)
            image2 = self.transform(img2)
            return [image1, image2], label, path, method
        else:
            with open(path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
            image = self.transform(img)
            return image, label, path, method


def set_env(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    set_gpu('0')

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x

def load_pretrain(model, pth_path):
    model_dict = model.state_dict()        
    pretrained_dict = torch.load(pth_path)['params']
    pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)