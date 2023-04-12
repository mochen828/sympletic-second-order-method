from __future__ import print_function
import os, time
import torch
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.datasets as torch_data
from nets import ConvODENet
from trainer import TrainerBase
import argparse
import logging
import time
import numpy as np
import util, options
import easydict
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import SGD, Adam
from torch_symplectic_adjoint import  odeint_adjoint as odesolve
from snopt import SNOpt, ODEFuncBase, ODEBlock

import colored_traceback.always
from ipdb import set_trace as debug


def build_optim_and_precond(opt, network):
    # build optimizer
    optim_dict = {"lr": opt.lr, 'weight_decay':opt.l2_norm, 'momentum':opt.momentum}
    if opt.optimizer =='Adam': optim_dict.pop('momentum', None)
    optim = SGD(network.parameters(), **optim_dict)

    # build precond
    if opt.optimizer=='SNOpt':
        kwargs = dict(eps=opt.snopt_eps, update_freq=opt.snopt_freq, full_precond=True)
        precond = SNOpt(network, **kwargs)
    else:
        precond = None

    return optim, precond

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ConcatConv2d(torch.nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ConvODEfunc(ODEFuncBase):
    def __init__(self, opt, hidden):
        super(ConvODEfunc, self).__init__(opt)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(hidden, hidden, 3, 1, 1)
        self.conv2 = ConcatConv2d(hidden, hidden, 3, 1, 1)

    def F(self, t, x):
        self.nfe += 1
        out = x
        out = self.conv1(t, out)
        out = self.relu(out)
        out = self.conv2(t, out)
        return out


class Trainer(TrainerBase):
    def __init__(self, train_loader, test_loader, network, optim, loss,
            precond=None, sched=None):
        super(Trainer, self).__init__(
            train_loader, test_loader, network, optim, loss, precond, sched
        )

    def prepare_var(self, opt, batch):
        var = easydict.EasyDict()
        [var.data, var.target] = [v.to(opt.device) for v in batch]
        return var

def build_clf_neural_ode(opt, hidden=64, t0=0.0, t1=1.0):
    odefunc = ConvODEfunc(opt, hidden)
    integration_time = torch.tensor([t0, t1]).float()
    ode = ODEBlock(opt, odefunc, odesolve, integration_time, is_clf_problem=True)
    network = ConvODENet(ode, hidden, input_dim=1).to(opt.device)

    print(network)
    print(util.magenta("Number of trainable parameters: {}".format(
        util.count_parameters(network)
    )))
    return network


def get_img_loader(opt, test_batch_size=1000):
    
    dataset_builder, root, input_dim, output_dim = {
        'mnist':   [torch_data.MNIST,  'data/img/mnist',  [1,28,28], 10],
        'SVHN':    [torch_data.SVHN,   'data/img/svhn',   [3,32,32], 10],
        'cifar10': [torch_data.CIFAR10,'data/img/cifar10',[3,32,32], 10],
    }.get('SVHN')
    opt.input_dim = input_dim
    opt.output_dim = output_dim

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    feed_dict = dict(download=True, root=root, transform=transform)
    train_dataset = dataset_builder(**feed_dict) if opt.problem=='SVHN' else dataset_builder(train=True, **feed_dict)
    test_dataset  = dataset_builder(**feed_dict) if opt.problem=='SVHN' else dataset_builder(train=False, **feed_dict)

    feed_dict = dict(num_workers=2, drop_last=True)
    train_loader = DataLoaderX(train_dataset, batch_size=opt.batch_size, shuffle=True, **feed_dict)
    test_loader  = DataLoaderX(test_dataset, batch_size=test_batch_size, shuffle=False, **feed_dict)

    return train_loader, test_loader

if __name__ == '__main__':

    # build opt and trainer
    opt = options.set()
    train_loader, test_loader = get_img_loader(opt)
    network = build_clf_neural_ode(opt, t1=opt.t1)
    optim, precond = build_optim_and_precond(opt, network)

    network = build_clf_neural_ode(opt, t1=opt.t1)
    optim, precond = build_optim_and_precond(opt, network)
    loss = F.cross_entropy
    trainer = Trainer(train_loader, test_loader, network, optim, loss, precond=precond)
    trainer.restore_checkpoint(opt, keys=["network","optim"])

    # save path
    os.makedirs(opt.result_dir, exist_ok=True)
    path = "{}/{}-{}_seed_{}_".format(opt.result_dir, opt.problem, opt.optimizer_config, opt.seed)

    # things we're going to collect over training
    losses       = util.Collector(path + 'train')
    eval_losses  = util.Collector(path + 'eval')
    accuracies   = util.Collector(path + 'accuracy')
    train_clocks = util.Collector(path + 'train_clock')
    eval_clocks  = util.Collector(path + 'eval_clock')
    if opt.use_adaptive_t1: t1s = util.Collector(path + 't1s')
    # strat training
    print(util.yellow("======= TRAINING START ======="))
    print(util.green(path))
    per_batch=len(trainer.train_loader)
    test_set=[]
    train_set=[]
    trainer.time_start()
    timeq=[]
    end=time.time()
    for ep in range(opt.epoch):
        for it, batch in enumerate(trainer.train_loader):
            train_it = ep*per_batch+it


            loss = trainer.train_step(opt, train_it, batch=batch)
            # util.print_train_progress(opt, trainer, train_it, loss)

            losses.append(loss)
            train_clocks.append(trainer.clock)
            if opt.use_adaptive_t1: t1s.append(trainer.get_ode_t1())

            if (train_it+1)%per_batch==0:
                eval_loss, accuracy=trainer.evaluate(opt, ep, train_it)
                util.print_eval_progress(opt, trainer, train_it //per_batch, eval_loss, accuracy=accuracy)
                aq=time.time()-end
                eval_losses.append(eval_loss)
                test_set.append(accuracy)
                
                accuracies.append(accuracy)
                eval_clocks.append(trainer.clock)
                timeq.append(aq)
        losses.save()
        eval_losses.save()
        accuracies.save()

        train_clocks.save()
        eval_clocks.save()
        if opt.use_adaptive_t1: t1s.save()

    time.sleep(1)
    print(util.yellow("======= TRAINING DONE ======="))
    pp=range(opt.epoch) 
    df=pd.DataFrame({'epoch':pp,'test_acc':test_set,'time':timeq})
    df.to_csv("./mydata/SS_SVHN.csv",index=False)
plt.title("Second Neural ODE")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(ymax=1,ymin=0)
plt.plot(pp,test_set,label="test_set")
plt.legend()
plt.savefig("./fig/SS_SVHN.eps")
plt.show()

    
