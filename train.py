from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from network import RAFTGMA

from utils import flow_viz
import datasets
import evaluate

from torch.cuda.amp import GradScaler
from loss_calculator import LossCalculatior

# exclude extremly large displacements
MAX_FLOW = 400


def convert_flow_to_image(image1, flow):
    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)
    flow_image = cv2.resize(flow_image, (image1.shape[3], image1.shape[2]))
    return flow_image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_loss(loss, loss_dict, name):
        if name not in loss_dict.keys():
            pass
        elif loss_dict[name] is None:
            pass
        else:
            this_loss = loss_dict[name].mean()
            #self.error_meter.update(name=name, val=this_loss.item(), num=batch_N, short_name=short_name)
            loss = loss + this_loss
        return loss

def compute_loss(loss_dict):
        loss = 0
        loss = fetch_loss(loss=loss, loss_dict=loss_dict, name='photo_loss')
        loss = fetch_loss(loss=loss, loss_dict=loss_dict, name='smooth_loss')
        loss = fetch_loss(loss=loss, loss_dict=loss_dict, name='census_loss')
        # photo_loss, smooth_loss, census_loss = output_dict['photo_loss'].mean(), output_dict['smooth_loss'], output_dict['census_loss']
        loss = fetch_loss(loss=loss, loss_dict=loss_dict, name='msd_loss')
        loss = fetch_loss(loss=loss, loss_dict=loss_dict, name='eq_loss')
        loss = fetch_loss(loss=loss, loss_dict=loss_dict, name='oi_loss')
        return loss

def self_supervised_loss(output_dict, valid):
    i_loss = (output_dict["predictions_f"][0] - output_dict["predictions_f"][-1]).abs()
    flow_loss = (valid[:, None] * i_loss).mean()
    return flow_loss

def sequence_loss2(output_dict, flow_gt, valid, gamma):
    n_predictions = len(output_dict['predictions_f'])
    predictions_f = output_dict['predictions_f']
    predictions_b = output_dict['predictions_b']
    image1 = output_dict['image1']
    image2 = output_dict['image2']
    occ_fw = output_dict['occ_fw']
    occ_bw = output_dict['occ_bw']
    loss_calculator = LossCalculatior()
    
        #  ===========SEQUENCE LOSS=======================================
    output_dict['losses'] = []
    flow_loss = 0.0 
    ss_weight = 0.4
    # exclude invalid pixels and extremely large displacements
    valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < MAX_FLOW)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        # CALCULATE LOSSES
        i_loss = loss_calculator(output_dict, image1, image2, predictions_f[i], predictions_b[i])
        i_loss = compute_loss(i_loss)
        flow_loss += i_weight * (i_loss ).mean()

    #custom self supervision
    #ss_loss = self_supervised_loss(output_dict, valid) * ss_weight
    #flow_loss = flow_loss*(1- ss_weight) + ss_loss
    # Calculate EPE - USED ONLY FOR MONITORING PURPOSES, IS NOT USED FOR TRAINING
    
    epe = torch.sum((output_dict["flow_f"]- flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    #print(flow_loss, len(output_dict["flow_f"]))

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'loss_total': flow_loss.cpu().detach().numpy()
    }
    #print(flow_loss)
    return flow_loss, metrics

def sequence_loss(output_dict, flow_gt, valid, gamma):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(output_dict['losses'])    
    flow_loss = 0.0 
    ss_weight = 0.0
    # exclude invalid pixels and extremely large displacements
    valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < MAX_FLOW)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        #i_loss = (flow_preds[i] - flow_gt).abs()
        i_loss = compute_loss(output_dict['losses'][i])
        flow_loss += i_weight * (i_loss ).mean()

    #flow_loss = flow_loss.deta
    #custom self supervision
    #ss_loss = self_supervised_loss(output_dict, valid)
    #flow_loss = flow_loss*(1- ss_weight) + ss_loss*ss_weight
    # Calculate EPE - USED ONLY FOR MONITORING PURPOSES, IS NOT USED FOR TRAINING
    
    epe = torch.sum((output_dict["flow_f"]- flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    #print(flow_loss, len(output_dict["flow_f"]))

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'loss_total': flow_loss.cpu().detach().numpy()
    }
    #print(flow_loss)
    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

from datetime import datetime
class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"

        

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_time = f"{current_time:>19}"
        print(training_str + metrics_str + time_left_hms + current_time + " " +  str(metrics_data[-1]))

        # logging running loss to total loss
        self.train_epe_list.append(np.mean(self.running_loss_dict['epe']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []

            self.running_loss_dict[key].append(metrics[key])

        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            self._print_training_status()
            self.running_loss_dict = {}


def main(args):

    model = nn.DataParallel(RAFTGMA(args), device_ids=args.gpus)

    print(f"Parameter Count: {count_parameters(model)}")

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    # if args.stage != 'chairs':
    #     model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args)

    while logger.total_steps <= args.num_steps:
        train(model, train_loader, optimizer, scheduler, logger, scaler, args)
        if logger.total_steps >= args.num_steps:
            plot_train(logger, args)
            plot_val(logger, args)
            break

    PATH = args.output+f'/{args.name}.pth'
    torch.save(model.state_dict(), PATH)
    return PATH

import time




def train(model, train_loader, optimizer, scheduler, logger, scaler, args):
    for i_batch, data_blob in enumerate(train_loader):
        tic = time.time()
        image1, image2, flow, valid = [x.cuda() for x in data_blob] # TODO: only getch im1 and im2
        #tac = time.time()
        
        optimizer.zero_grad()
        
        output_dict = model(image1, image2)
    
        loss, metrics = sequence_loss(output_dict, flow, valid, args.gamma)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        toc = time.time()
        #print(tac - tic, toc - tac)
        metrics['time'] = toc - tic
        logger.push(metrics)

        # Validate
        if logger.total_steps % args.val_freq == args.val_freq - 1:
            validate(model, args, logger)
            plot_train(logger, args)
            plot_val(logger, args)
            PATH = args.output + f'/{logger.total_steps+1}_{args.name}.pth'
            torch.save(model.state_dict(), PATH)

        if logger.total_steps >= args.num_steps:
            break


def validate(model, args, logger):
    model.eval()
    results = {}

    # Evaluate results
    for val_dataset in args.validation:
        if val_dataset == 'chairs':
            results.update(evaluate.validate_chairs(model.module, args.iters))
        elif val_dataset == 'sintel':
            results.update(evaluate.validate_sintel(model.module, args.iters))
        elif val_dataset == 'kitti':
            results.update(evaluate.validate_kitti(model.module, args.iters))

    # Record results in logger
    for key in results.keys():
        if key not in logger.val_results_dict.keys():
            logger.val_results_dict[key] = []
        logger.val_results_dict[key].append(results[key])

    logger.val_steps_list.append(logger.total_steps)
    model.train()


def plot_val(logger, args):
    for key in logger.val_results_dict.keys():
        # plot validation curve
        plt.figure()
        plt.plot(logger.val_steps_list, logger.val_results_dict[key])
        plt.xlabel('x_steps')
        plt.ylabel(key)
        plt.title(f'Results for {key} for the validation set')
        plt.savefig(args.output+f"/{key}.png", bbox_inches='tight')
        plt.close()
        f = open(args.output+f"/{key}.txt", "w")
        f.write(str(logger.val_steps_list))
        f.write(str(logger.val_results_dict[key]))
        f.close()


def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_epe_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output+"/train_epe.png", bbox_inches='tight')
    plt.close()
    f = open(args.output+"/train_epe.txt", "w")
    f.write(str(logger.train_steps_list))
    f.write(str(logger.train_epe_list))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output', type=str, default='checkpoints', help='output directory to save checkpoints and plots')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--model_name', default='', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    main(args)
