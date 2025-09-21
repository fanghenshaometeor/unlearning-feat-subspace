import copy
import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim

import arg_parser

import utils
from utils import setup_seed
from utils import Logger
from utils import save_checkpoint
from utils_data import setup_model
from utils_data import tinyimgnet_dataloaders
from utils_data import create_retain_forget_partition
from utils_data import dataset_convert_to_test

args = arg_parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)
args.logs_path = os.path.join(os.path.join(args.save_dir, "train.log"))
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        image, target = image.cuda(), target.cuda()

        # compute output
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.float().data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
        
        if (i+1) == len(train_loader):
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def main():
    setup_seed(args.seed)

    # ======== create models and dataloaders ========
    if args.dataset == "tinyimagenet":
        args.num_classes = 200
        train_loader, test_loader = tinyimgnet_dataloaders(args)
    

    print()
    print(f"dataset & model: {args.dataset} & {args.arch}")
    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of test  dataset {len(test_loader.dataset)}")

    model = setup_model(args)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_time_total = time.time()
    for epoch in range(args.epochs):
        start_time = time.time()
        print()
        print(
            "Epoch #{}/{}, Learning rate: {}".format(
                epoch+1, args.epochs, optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        train(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step()
        print("one epoch duration:{}".format(time.time() - start_time))
    print("Total duration:{}".format(time.time() - start_time_total)) 

    save_checkpoint(
        {   "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        save_path=args.save_dir,
        filename="ckpt_pretrain.pth.tar"
    )


if __name__ == "__main__":
    main()