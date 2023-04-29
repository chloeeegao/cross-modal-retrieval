import torch
import pickle
import os
import random
random.seed(1234)
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import torchvision.transforms as transforms
from info_extraction import info_extract
from utils.utils import get_token_ids, list2Tensors

class Recipe1M(Dataset):

    def __init__(self,root, transform=None, split='train'):

        self.data = pickle.load(open(os.path.join(root,'traindata', split + '.pkl'), 'rb'))
        self.root = root
        self.transform = transform
        self.split = split
        self.ids = list(self.data.keys())
        self.max_ingrs = 20
        # self.max_instrs = 20
        self.max_seq_len = 15

        self.vocab_inv = pickle.load(open(os.path.join(root,'vocab.pkl'),'rb'))
        self.vocab = {}
        for k, v in self.vocab_inv.items():
            if type(v) != str:
                v = v[0]
            self.vocab[v] = k
        
        self.index_mapper = dict()
        if self.split == 'train':
            j = 0
            for i, id in enumerate(self.ids):
                img_list = self.data[id]['images']
                if len(img_list)<=5:
                    for img in img_list:
                        self.index_mapper[j] = (id, img)
                        j += 1
                else:
                    # random choose 5 images from img_list
                    random_choie = random.sample(img_list, 5)
                    for img in random_choie:
                        self.index_mapper[j] = (id, img)
                        j += 1
        else:
            for i, id in enumerate(self.ids):
                img_list = self.data[id]['images']
                # random choose 1 image from img_list
                img = random.choice(img_list)
                self.index_mapper[i] = (id, img)
        
    def __len__(self):
        return len(self.index_mapper)

    def get_ids(self):
        return self.ids

    def get_vocab(self):
        try: 
            return self.vocab_inv
        except:
            return None

    def __getitem__(self, index):
        
        recipe_id, img_name = self.index_mapper[index]
        entry = self.data[recipe_id]

        img_name = '/'.join(img_name[:4])+'/'+img_name
        img = Image.open(os.path.join(self.root, self.split, img_name))
        if self.transform is not None:
            img = self.transform(img)
            
        title = entry['title']
        ingrs = entry['ingredients'][:self.max_ingrs]
        # instrs = entry['instructions'][:self.max_instrs]
        
        action_ing = info_extract(recipe_id, entry)[:self.max_ingrs]
        
        title = torch.Tensor(get_token_ids(title, self.vocab)[:self.max_seq_len])
        if len(action_ing) != 0:
            extract_info = list2Tensors([get_token_ids(' '.join(sent), self.vocab)[:self.max_seq_len] for sent in action_ing])
        else:
            extract_info = list2Tensors([get_token_ids(ing, self.vocab)[:self.max_seq_len] for ing in ingrs])
        
        return recipe_id, title, extract_info, img


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

    ids, title, act_ing, image = zip(*data)

    if image[0] is not None:
        image = torch.stack(image, 0)
    else:
        image = None

    title = pad_input(title)
    act_ing = pad_input(act_ing)
    
    return ids, title, act_ing, image

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

