import copy
import glob
import os
from shutil import move

import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from tqdm import tqdm

import models
from models import *

def cifar_dataloaders(args):

    if args.dataset == 'cifar10':
        transform_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    elif args.dataset == 'cifar100':
        transform_normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
    else:
        raise ValueError("Dataset not supprot yet !")


    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transform_normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms_normalize
    ])

    if args.dataset == 'cifar10':
        train_set = CIFAR10(args.data_dir, train=True, transform=train_transform, download=True)
        test_set = CIFAR10(args.data_dir, train=False, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        train_set = CIFAR100(args.data_dir, train=True, transform=train_transform, download=True)
        test_set = CIFAR100(args.data_dir, train=False, transform=test_transform, download=True)
    else:
        raise ValueError("Dataset not supprot yet !")

    loader_args = {"num_workers": 8, "pin_memory": True}

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)

    return train_loader, test_loader


def tinyimgnet_dataloaders(args):
    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = ImageFolder(os.path.join(args.data_dir,'train'),transform=train_transform)
    test_set = ImageFolder(os.path.join(args.data_dir,'val'),transform=test_transform)

    loader_args = {"num_workers": 8, "pin_memory": True}

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)

    return train_loader, test_loader

def imagenet_dataloaders(args):


    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),      # Random crop and resize to 224x224
        transforms.RandomHorizontalFlip(),      # Random horizontal flip
        transforms.ToTensor(),                  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = ImageFolder(os.path.join(args.data_dir,'train'),transform=train_transform)
    test_set = ImageFolder(os.path.join(args.data_dir,'val'),transform=test_transform)

    loader_args = {"num_workers": 8, "pin_memory": True}

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)

    return train_loader, test_loader


def rafdb_dataloaders(args):

    train_transform = transforms.Compose([
        transforms.Resize(100),
        transforms.RandomCrop(100, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
    ])

    train_set = ImageFolder(os.path.join(args.data_dir,'aligned_train'),transform=train_transform)
    test_set = ImageFolder(os.path.join(args.data_dir,'aligned_test'),transform=test_transform)

    loader_args = {"num_workers": 8, "pin_memory": True}

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)

    return train_loader, test_loader

class vggface2_dataset(Dataset):
    def __init__(self, config, transform, train=None):
        if train==True:
            self.samples = config["train"]
        else:
            self.samples = config["test"]
        self.targets = [s[1] for s in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, _ = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = self.targets[index]
        return img, label

def vggface2_dataloaders(args):

    train_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.RandomCrop(112, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
    ])

    config_path = "/path/to/config/vggface2_200ids_config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        print(f"Load vggface2 data config from {config_path}")
    train_set = vggface2_dataset(config,transform=train_transform,train=True)
    test_set = vggface2_dataset(config,transform=test_transform,train=False)

    loader_args = {"num_workers": 8, "pin_memory": True}

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)

    return train_loader, test_loader

def create_retain_forget_partition(train_set, args):
    if args.class_forget and args.random_forget:
        raise ValueError("Only one of `args.class_forget` and `args.random_forget` can be activated")

    if args.random_forget:
        num_to_forget = args.num_to_forget
        if num_to_forget is None:
            raise ValueError("Must specify the number of forget data if `args.random_forget` is activated.")        
        else:
            assert num_to_forget <= len(train_set), f"Want to replace {num_to_forget} indexes but only {len(train_set)} samples in dataset" 
            
        forget_idx = np.flatnonzero(np.ones_like(train_set.targets))
        rng = np.random.RandomState(args.seed)
        forget_idx = rng.choice(forget_idx, size=num_to_forget, replace=False)
        print()
        print(f"Randomly selecting {num_to_forget} from {len(train_set)} images as `FORGET` data.")
        print(f"Forgetting indexes {forget_idx}.")
        
        retain_idx = list(set(range(len(train_set))) - set(forget_idx))
        forget_set = Subset(train_set, forget_idx)
        retain_set = Subset(train_set, retain_idx)
    
    if args.class_forget and args.label_to_forget is not None:
        label_to_forget = [int(val) for val in args.label_to_forget]
        train_idx = np.arange(len(train_set))
        forget_idx = np.array([],dtype=int)
        for label in label_to_forget:            
            forget_idx = np.concatenate((forget_idx, train_idx[(np.array(train_set.targets) == label)]))
        
        print()
        print(f"Labels to be forgetten: {label_to_forget}")
        print(f"Forgetting indexes {forget_idx}.")


        retain_idx = list(set(range(len(train_set))) - set(forget_idx))
        forget_set = Subset(train_set, forget_idx)
        retain_set = Subset(train_set, retain_idx)
    

    print()
    print(f"number of retain dataset {len(retain_set)}")
    print(f"number of forget dataset {len(forget_set)}")

    loader_args = {"num_workers": 8, "pin_memory": True}

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    forget_loader = DataLoader(forget_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)
    retain_loader = DataLoader(retain_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)

    return forget_loader, retain_loader

def create_retain_forget_partition_continual(train_set, current_label_to_forget, total_label_to_forget, args):
    if args.random_forget:
        raise ValueError("`args.random_forget` cannot be activated")
    
    if args.class_forget:
        label_to_forget = [int(val) for val in args.label_to_forget]
        label_forgotten = np.setdiff1d(total_label_to_forget, current_label_to_forget)
        label_retain = np.setdiff1d(np.arange(args.num_classes), total_label_to_forget)

        train_idx = np.arange(len(train_set))
        current_forget_idx = np.array([],dtype=int)
        previos_forget_idx = np.array([],dtype=int)
        retain_idx = np.array([],dtype=int)
        for label in label_to_forget:            
            current_forget_idx = np.concatenate((current_forget_idx, train_idx[(np.array(train_set.targets) == label)]))
        for label in label_forgotten:
            previos_forget_idx = np.concatenate((previos_forget_idx, train_idx[(np.array(train_set.targets) == label)]))
        for label in label_retain:
            retain_idx = np.concatenate((retain_idx, train_idx[(np.array(train_set.targets) == label)]))

        current_forget_set = Subset(train_set, current_forget_idx)
        previos_forget_set = Subset(train_set, previos_forget_idx)
        retain_set = Subset(train_set, retain_idx)
    

        print()
        print(f"number of forget    dataset {len(current_forget_set)}")
        print(f"number of forgetten dataset {len(previos_forget_set)}")
        print(f"number of retain    dataset {len(retain_set)}")

    loader_args = {"num_workers": 8, "pin_memory": True}

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    forget_loader = DataLoader(current_forget_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)
    forgotten_loader = None if previos_forget_idx.shape[0] == 0 else DataLoader(previos_forget_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)
    retain_loader = None if retain_idx.shape[0] == 0 else DataLoader(retain_set, batch_size=args.batch_size, shuffle=True, worker_init_fn=_init_fn if args.seed is not None else None, **loader_args)

    return forgotten_loader, forget_loader, retain_loader


def dataset_convert_to_test(train_loader, args):
    if args.dataset == "tinyimagenet":
        test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if args.dataset == "imagenet":
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    if args.dataset == "rafdb":
        test_transform = transforms.Compose([
            transforms.Resize(100),
            transforms.ToTensor(),
        ])
    
    if args.dataset == "vggface2":
        test_transform = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
        ])

    train_loader.dataset.transform = test_transform
    return







def setup_model(args):

    if args.arch == "swin_t" and args.dataset == "tinyimagenet":
        if args.unlearn == "subspace":
            model = model_dict[args.arch](weights="DEFAULT",s=args.subspace_s)
        else:
            model = model_dict[args.arch](weights="DEFAULT")
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, args.num_classes) if args.num_classes > 0 else nn.Identity()

    elif args.arch == "resnet50" and args.dataset == "imagenet":
        if args.unlearn == "subspace":
            model = model_dict[args.arch](pretrained=True,s=args.subspace_s)
        else:
            model = model_dict[args.arch](pretrained=True)

    elif args.arch == "resnet18" and args.dataset == "rafdb":
        if args.unlearn == "subspace":
            model = model_dict[args.arch](num_classes=args.num_classes, s=args.subspace_s)
        else:
            model = model_dict[args.arch](num_classes=args.num_classes)
    
    elif args.arch == "resnet50" and args.dataset == "vggface2":
        if args.unlearn == "subspace":
            model = model_dict[args.arch](num_classes=args.num_classes, s=args.subspace_s)
        else:
            model = model_dict[args.arch](num_classes=args.num_classes)

    return model




















# if __name__ == "__main__":
#     import arg_parser
#     args = arg_parser.parse_args()
#     args.dataset = 'cifar10'
#     args.data_dir = '~/data/cifar10'
#     args.seed = 0
#     args.batch_size = 256
#     train_loader, val_loader, test_loader = cifar10_dataloaders(args)

#     args.class_forget = True
#     args.label_to_forget = ['1','3']
#     create_retain_forget_partition(train_loader.dataset, args)