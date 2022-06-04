from subprocess import list2cmdline
import torch
import pickle
from cooking_action import extract_action
import os
import random
from utiliz import get_token_ids, list2Tensors
random.seed(1234)
from random import choice
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import torchvision.transforms as transforms
import numpy as np



class Recipe1M(Dataset):

    def __init__(self,root,transform=None, split='train'):

        self.data = pickle.load(open(os.path.join(root,'traindata', split + '.pkl'), 'rb'))
        self.valid_ingrd = pickle.load(open(os.path.join(root,'valid_ingredients.pkl'),'rb'))
        self.vocab_inv = pickle.load(open('/data/s2478846/scripts/data/vocab.pkl','rb'))
        self.vocab = {}

        for k, v in self.vocab_inv.items():
            if type(v) != str:
                v = v[0]
            self.vocab[v] = k

        self.root = root
        self.transform = transform
        self.split = split
        self.ids = list(self.data.keys())

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        return self.ids

    def get_vocab(self):
        try: 
            return self.vocab_inv
        except:
            return None

    def __getitem__(self, id):

        entry = self.data[self.ids[id]]

        if self.split == 'train':
            img_name = choice(entry['images'])

        else:
            img_name = entry['images'][0]

        img_name = '/'.join(img_name[:4])+'/'+img_name
        img = Image.open(os.path.join(self.root, self.split, img_name))
        if self.transform is not None:
            img = self.transform(img)

        # title = entry['title']
        ingrs = self.valid_ingrd[self.ids[id]]
        instrs = entry['instructions']

        # extract action and corresponding ingredients
    
        action_ingrs = extract_action(ingrs, instrs)
        recipe = list2Tensors(get_token_ids(action_ingrs, self.vocab))

        return self.ids[id], recipe, img


def pad_input(input):
    """
    creates a padded tensor to fit the longest sequence in the batch
    """
    if len(input[0].size()) == 1:
        l = [len(elem) for elem in input]
        targets = torch.zeros(len(input), max(l)).long()
        for i, elem in enumerate(input):
            end = l[i]
            targets[i, :end] = elem[:end]
    else:
        n, l = [], []
        for elem in input:
            n.append(elem.size(0))
            l.append(elem.size(1))
        targets = torch.zeros(len(input), max(n), max(l)).long()
        for i, elem in enumerate(input):
            targets[i, :n[i], :l[i]] = elem
    return targets


def collate_fn(data):

    ids, recipe, image = zip(*data)

    if image[0] is not None:
        image = torch.stack(image, 0)
    else:
        image = None

    recipe_target = pad_input(recipe)

    return ids, recipe_target, image

def get_loader(root, batch_size, resize, im_size, split='train', 
                mode='train', augment=True, drop_last=True):

    transforms_list = [transforms.Resize((resize))]
    if mode == 'train' and augment:
        # Image preprocessing, normalization for pretrained resnet
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomCrop(im_size))

    else:
        transforms_list.append(transforms.CenterCrop(im_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))

    transforms_ = transforms.Compose(transforms_list)

    ds = Recipe1M(root, transform=transforms_, split=split)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=multiprocessing.cpu_count(),
                        collate_fn=collate_fn, drop_last=drop_last)

    return loader, ds


