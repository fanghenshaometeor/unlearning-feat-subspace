import copy
import os
import sys
import time
import random
from collections import OrderedDict
from itertools import cycle

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, Subset

import numpy as np

import arg_parser
import utils
from utils import setup_seed
from utils import Logger
from utils import save_checkpoint
from utils_data import setup_model
from utils_data import tinyimgnet_dataloaders, imagenet_dataloaders
from utils_data import create_retain_forget_partition
from utils_data import dataset_convert_to_test

from mia_evaluation import get_membership_attack_prob, get_SVC_MIA

import geoopt
from geoopt.optim import RiemannianAdam

args = arg_parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)
if args.class_forget and args.label_to_forget is not None:
    log_file = "unlearn_class_"+"_".join(str(x) for x in args.label_to_forget)+".log"
if args.class_forget and args.num_label_to_forget is not None:
    log_file = "unlearn_class_"+str(args.seed)+"_"+str(args.num_label_to_forget)+".log"
if args.random_forget and args.num_to_forget is not None:
    log_file = "unlearn_random_"+str(args.seed)+"_"+str(args.num_to_forget)+".log"
args.logs_path = os.path.join(os.path.join(args.save_dir, log_file))
sys.stdout = Logger(filename=args.logs_path,stream=sys.stdout)

# one-shot feature fetching to calculate covariances
def extract_feat(data_loader, model, args):
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    if args.arch == "swin_t":
        featdim = model.head.in_features
    if args.arch == "resnet50":
        featdim = model.fc.in_features
    feat_log = np.zeros((len(data_loader.dataset), featdim))
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.cuda(), y.cuda()

            # compute output
            feat, output = model.forward_feat(x)
            output = output.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, y)[0]
            top1.update(prec1.item(), x.size(0))

            start_ind = batch_idx * args.batch_size
            end_ind = min((batch_idx + 1) * args.batch_size, len(data_loader.dataset))
            feat_log[start_ind:end_ind, :] = feat.data.cpu().numpy()

    return top1.avg, feat_log

# inference forward of f_U
def validate_rconst(data_loader, model, args):
    """
    Run evaluation
    """
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.cuda(), y.cuda()

            # compute output
            output = model.forward_U(x) # <-- key function
            output = output.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, y)[0]
            top1.update(prec1.item(), x.size(0))

    return top1.avg


# optimization of SUN
def subspace_learning(sigma_f, sigma_r, model, optimizer, args):

    losses = {}
    losses['trace-retain'] = utils.AverageMeter()
    losses['trace-forget'] = utils.AverageMeter()

    start = time.time()
    for step in range(args.subspace_steps):
        # J_rm
        loss_trace_retain = torch.trace(sigma_r - model.U @ model.U.t() @ sigma_r @ model.U @ model.U.t()) / torch.trace(sigma_r)
        # J_fg
        loss_trace_forget = torch.trace(model.U.t() @ sigma_f @ model.U) / torch.trace(sigma_f)
        loss = loss_trace_retain*loss_trace_retain + loss_trace_forget*loss_trace_forget

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses['trace-retain'].update(loss_trace_retain.float().item(), 1)
        losses['trace-forget'].update(loss_trace_forget.float().item(), 1)

        if step % 10 == 0 or (step+1) == args.subspace_steps:
            end = time.time()
            print(
                "Step: [{0}/{1}]\t"
                "Loss Trace Retain {loss1.val:.4f} ({loss1.avg:.4f})\t"
                "Loss Trace Forget {loss2.val:.4f} ({loss2.avg:.4f})\t"
                "Time {2:.2f}\t Learning rate {3}".format(
                    step+1, args.subspace_steps, end - start, optimizer.state_dict()["param_groups"][0]["lr"], loss1=losses['trace-retain'], loss2=losses['trace-forget']
                )
            )
            start = time.time()
    
    return

def main():
    print()
    setup_seed(args.seed)

    # ======== create models and dataloaders ========
    if args.dataset == "tinyimagenet":
        args.num_classes = 200
        train_loader, test_loader = tinyimgnet_dataloaders(args)
    elif args.dataset == "imagenet":
        args.num_classes = 1000
        train_loader, test_loader = imagenet_dataloaders(args)

    # this `if` works for extreme unlearning
    if args.class_forget and args.num_label_to_forget is not None:
        num_label_to_forget = args.num_label_to_forget
        assert num_label_to_forget <= args.num_classes, f"Want to unlearn {num_label_to_forget} classes but only {args.num_classes} classes in dataset"
        args.label_to_forget = np.random.choice(np.arange(args.num_classes), size=num_label_to_forget, replace=False)
    
    tr_forget_loader, tr_retain_loader = create_retain_forget_partition(train_loader.dataset, args)
    if args.class_forget:
        te_forget_loader, te_retain_loader = create_retain_forget_partition(test_loader.dataset, args)
    if args.random_forget:
        te_forget_loader, te_retain_loader = None, None
    
    unlearn_data_loaders = OrderedDict(
        train=train_loader, tr_retain=tr_retain_loader, tr_forget=tr_forget_loader,
        test=test_loader, te_retain=te_retain_loader, te_forget=te_forget_loader
    )

    model = setup_model(args)
    model = model.cuda()

    # load pre-trained checkpoints, only for tinyimagenet
    # for imagenet, directly use the pre-trained checkpoints released from PyTorch
    if args.dataset == "tinyimagenet":
        checkpoint = torch.load(args.model_path, map_location="cpu")
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=False)
        print()
        print(f"Load model from {args.model_path}")

    # ======== initialize Riemannian optimizers ========
    U_params, net_params = [], []
    for name, param in model.named_parameters():
        if name == 'U':
            U_params += [param]
        else:
            net_params += [param]
    optimizer = RiemannianAdam(params=U_params, lr=args.unlearn_lr, weight_decay=args.weight_decay)

    # ======== Use the test transform on training dataloaders to correctly extract features ========
    for name, loader in unlearn_data_loaders.items():
        if loader is not None:
            dataset_convert_to_test(loader, args)

    # ======== extract/load covaraince matrix of retain & forget data ========
    if args.class_forget and args.label_to_forget is not None:
        sigma_r_path = os.path.join(args.save_dir, "sigma_r_"+"_".join(str(x) for x in args.label_to_forget)+".npy")
        sigma_f_path = os.path.join(args.save_dir, "sigma_f_"+"_".join(str(x) for x in args.label_to_forget)+".npy")
    if args.class_forget and args.num_label_to_forget is not None:
        sigma_r_path = os.path.join(args.save_dir, "sigma_r_"+str(args.seed)+"_"+str(args.num_label_to_forget)+".npy")
        sigma_f_path = os.path.join(args.save_dir, "sigma_f_"+str(args.seed)+"_"+str(args.num_label_to_forget)+".npy")
    if args.random_forget and args.num_to_forget is not None:
        sigma_r_path = os.path.join(args.save_dir, "sigma_r_"+str(args.seed)+"_"+str(args.num_to_forget)+"_random.npy")
        sigma_f_path = os.path.join(args.save_dir, "sigma_f_"+str(args.seed)+"_"+str(args.num_to_forget)+"_random.npy")
    
    if os.path.exists(sigma_r_path):
        sigma_r = np.load(sigma_r_path,allow_pickle=True)
        print()
        print(f"Load covaraince matrix of retain data from {sigma_r_path}")
    else:
        print()
        print("Calculating covariance matrix of retain data...")
        acc_retain, feat_retain = extract_feat(tr_retain_loader, model, args)
        sigma_r = np.cov(feat_retain, rowvar=False)
        np.save(sigma_r_path, sigma_r)
        print(f"Covariance saved at {sigma_r_path} with training retain acc = {acc_retain}")
        print("feat.shape: ", feat_retain.shape, ", sigma_r.shape: ", sigma_r.shape)
    
    if os.path.exists(sigma_f_path):
        sigma_f = np.load(sigma_f_path,allow_pickle=True)
        print()
        print(f"Load covaraince matrix of forget data from {sigma_f_path}")
    else:
        print()
        print("Calculating covariance matrix of forget data...")
        acc_forget, feat_forget = extract_feat(tr_forget_loader, model, args)
        sigma_f = np.cov(feat_forget, rowvar=False)
        np.save(sigma_f_path, sigma_f)
        print(f"Covariance saved at {sigma_f_path} with training forget acc = {acc_forget}")
        print("feat.shape: ", feat_forget.shape, ", sigma_f.shape: ", sigma_f.shape)

    sigma_r = torch.from_numpy(sigma_r.astype(np.float32)).cuda()
    sigma_f = torch.from_numpy(sigma_f.astype(np.float32)).cuda()
    u, s, v = torch.linalg.svd(sigma_r)
    explained_variance_ratio = s**2 / torch.sum(s**2)
    print(f"For reference, evr[:s] with s={args.subspace_s}: ", torch.sum(explained_variance_ratio[:args.subspace_s]))


    # ======== subspace unlearning ========
    print()
    print(f"Method: Subspace, s={args.subspace_s}.")

    start_time = time.time()
    subspace_learning(sigma_f, sigma_r, model, optimizer, args)
    print("Total duration:{}".format(time.time() - start_time))  

    # ======== evaluation ========
    print()
    print(f"Validating {args.unlearn}...")
    evaluation_result = {}
    accuracy = {}
    for name, loader in unlearn_data_loaders.items():
        if args.dataset == "imagenet" and args.class_forget and name in ['train','test']:
            accuracy[name] = None # avoid redudant computations for the large-scale ImageNet-1K
            continue
        if loader is not None:
            val_acc = validate_rconst(loader, model, args)
        else:
            val_acc = None
        accuracy[name] = val_acc

        evaluation_result["accuracy"] = accuracy
    
    # mia_evaluation
    num_tr_retain = len(tr_retain_loader.dataset)
    num_test = len(test_loader.dataset)
    num_mia_train = min(num_tr_retain, num_test)
    print(f"{num_mia_train} random samples from tr_retain and test to train the bi-classifier for MIA")
    sample_tr_retain_set = Subset(tr_retain_loader.dataset, random.sample(range(num_tr_retain),num_mia_train))
    sample_test_set = Subset(test_loader.dataset, random.sample(range(num_test),num_mia_train))
    sample_tr_retain_loader = DataLoader(sample_tr_retain_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    sample_test_loader = DataLoader(sample_test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    mia_result = get_SVC_MIA(
        shadow_train=sample_tr_retain_loader, 
        shadow_test=sample_test_loader,
        target_train=None,
        target_test=tr_forget_loader,
        model=model,
        args=args
    )
    evaluation_result["mia"] = (1-mia_result["confidence"])*100

    print("Validation finished.")
    print(f"Train accuracy, full/retain/forget: {accuracy['train']}/{accuracy['tr_retain']}/{accuracy['tr_forget']}")
    print(f"Test  accuracy, full/retain/forget: {accuracy['test']}/{accuracy['te_retain']}/{accuracy['te_forget']}")
    print(f"MIA (confidence): {evaluation_result['mia']}")
    #
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    if args.class_forget and args.label_to_forget is not None:
        filename = "ckpt_class_"+"_".join(str(x) for x in args.label_to_forget)+".pth.tar"
    if args.class_forget and args.num_label_to_forget is not None:
        filename = "ckpt_class_"+str(args.seed)+"_"+str(args.num_label_to_forget)+".pth.tar"
    if args.random_forget and args.num_to_forget is not None:
        filename = "ckpt_random_"+str(args.seed)+"_"+str(args.num_to_forget)+".pth.tar"
    save_checkpoint(state=state, save_path=args.save_dir, filename=filename)



if __name__ == "__main__":
    main()