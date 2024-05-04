import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import ipdb
import json
import wandb
import socket
import tqdm
import argparse
from data.data_utils_feature import get_loader_feature
import models.lossnet as lossnet
from config import *

# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def write_one_results(path, json_data):
    with open(path, "w") as outfile:
        json.dump(json_data, outfile)   

def test(models, dataloaders, args):
     result = []
     with torch.no_grad():
        epoch_iterator = tqdm(dataloaders)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            features, target_loss = batch
            pred_loss = models(features)
            result.extend(pred_loss.cpu().tolist())
     return result

def save_model(args, model, epoch):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_checkpoint_{epoch}.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)

def train(args):
    ipdb.set_trace()
    train_loader, test_loader, train_datasets, test_datasets = get_loader_feature(args)

    loss_module = lossnet.LossNet().to(args.device)
    num_epochs = args.num_epochs
    optim_module   = optim.SGD(loss_module.parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
    sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

    for epoch in range(num_epochs):
        sched_module.step()

        for epoch in range(num_epochs):
            loss_module.train()
            for data in tqdm(train_loader, leave=False, total=len(train_loader)):
                features = data[0]
                target_loss = data[1]
                pred_loss = loss_module(features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
                loss.backward()
                optim_module.step()

        if True:
            result = test(loss_module, test_loader, args)
            save_model(args, loss_module, epoch)
            path = os.path.join(args.output_dir, f"{args.name}_image_losses_{epoch}.json")
            json_objects = {"losses": result}
            write_one_results(path, json_objects)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument('--enable_wandb', action='store_true',
                        help="Whether to enable wandb")
    parser.add_argument("--data_dir", type=str, default="../orkspace/DINO/data/5_scale_31/",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="num epochs.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
     # start a new wandb run to track this script
    if args.enable_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Bosch_active_testing",
            config=args,
            entity="susanbao",
            notes=socket.gethostname(),
            name=args.name,
            job_type="training"
        )
    # ipdb.set_trace()

    # Training
    train(args)
    if args.enable_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
