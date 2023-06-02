import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from dataset import Recipe1M
from module.model import RecipeVL
from config import get_args
from Trainer import Trainer
from transformers import BertTokenizer
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def ddp_setup(rank, world_size):
    # random_num = str(random.randint(12345,12399))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12088'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
def get_dataset(args, tokenizer, sample, split='train', 
            mode='train', augment=True):       
    transforms_list = [transforms.Resize((args.resize))]
    if mode == 'train' and augment:
        # Image preprocessing, normalization for pretrained resnet
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomCrop(args.im_size))
    else:
        transforms_list.append(transforms.CenterCrop(args.im_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))
    transforms_ = transforms.Compose(transforms_list)
    dataset = Recipe1M(args, tokenizer=tokenizer, transforms=transforms_, split=split, sample=sample)
    return dataset 

def main(rank, args, world_size):
    mp.set_start_method('fork', force=True)
    ddp_setup(rank, world_size)
    set_seed(seed=args.seed)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case='uncased')
    loss_names = {'itm': 1, 'mlm': 1, 'irtr': 1}
    model = RecipeVL(args, loss_names)
    
    if args.do_train:
        train_dataset = get_dataset(args, tokenizer, sample='all')
        test_dataset = get_dataset(args, tokenizer, split='val', mode='val', sample=args.eval_sample)
        trainer = Trainer(args, model, train_dataset, test_dataset, rank, is_train=True)
        trainer.train()
    elif args.do_eval:
        trainer = Trainer(args, model, rank=rank, is_train=False)
        test_dataset = get_dataset(args, tokenizer, split='test', mode = 'eval', sample='all')
        trainer.eval(test_dataset, tokenizer, N=args.eval_sample, K=args.eval_times)
            
    destroy_process_group()

if __name__ == "__main__":
    args = get_args()  
    world_size = torch.cuda.device_count()
    args.n_gpus = world_size
    mp.spawn(main, args=(args, world_size), nprocs=world_size, join=True)
    
    