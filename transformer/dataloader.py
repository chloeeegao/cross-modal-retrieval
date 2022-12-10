import torch
import pickle
import os
import random
from utils import get_token_ids, list2Tensors
random.seed(1234)
from random import choice
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import torchvision.transforms as transforms
import spacy 
from transformers import BertTokenizer
from build_vocab import Vocabulary

nlp= spacy.load('en_core_web_sm')
try: 
    ruler = nlp.add_pipe("entity_ruler")
except ValueError:
    ruler = nlp.add_pipe(nlp.create_pipe("entity_ruler"))
ruler.from_disk("/data/s2478846/git_repo/cross-modal-retrieval/data/patterns.jsonl")


def info_extract(instrs):
    sents = ['I ' + instr for instr in instrs]
    action_dict ={}
    for sent in sents:
        doc = nlp(sent)
        ingr_entities = [ent for ent in doc.ents if ent.label_=='ING']
        root = [token.lemma_ for token in doc if token.dep_ =='ROOT']
        if len(ingr_entities) > 0:
            for ent in ingr_entities:
                head = ent.root.head
                ingr = ent.text
                if head.pos_ == "VERB":
                    if head.lemma_ in list(action_dict.keys()) and ingr not in action_dict[head.lemma_]:
                        action_dict[head.lemma_].append(ingr)
                    else:
                        action_dict[head.lemma_] = [ingr]
                else:
                    if root[0] in list(action_dict.keys()) and ingr not in action_dict[root[0]]:
                        action_dict[root[0]].append(ingr)
                    else:
                        action_dict[root[0]] = [ingr]
    return action_dict



class Recipe1M(Dataset):

    def __init__(self,root, transform=None, split='train', initial='scratch'):

        self.data = pickle.load(open(os.path.join(root,'traindata', split + '.pkl'), 'rb'))
        # self.valid_ingrd = pickle.load(open(os.path.join(root,'valid_ingredients.pkl'),'rb'))
        self.root = root
        self.transform = transform
        self.split = split
        self.ids = list(self.data.keys())
        self.max_text_len = 50
        self.max_ingrs = 20
        self.max_instrs = 20
        self.max_action = 8
        self.initial = initial
        
        if self.initial=='pretrain':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case='uncased')
        else:
            self.vocab_inv = pickle.load(open(os.path.join(root,'vocab.pkl'),'rb'))
            # vocab_class = pickle.load(open('../data/recipe1m_vocab_toks.pkl','rb'))
            # self.vocab = vocab_class.idx2word
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

        # if self.split == 'train':
        #     img_name = choice(entry['images'])
        # else:
        #     img_name = entry['images'][0]

        img_name = '/'.join(img_name[:4])+'/'+img_name
        img = Image.open(os.path.join(self.root, self.split, img_name))
        if self.transform is not None:
            img = self.transform(img)
            
        title = entry['title']
        ingrs = entry['ingredients'][:self.max_ingrs]
        instrs = entry['instructions'][:self.max_instrs]
        
        action_dict = info_extract(instrs)
        
        # if len(action_dict) == 0:
        #     action_dict = info_extract(ingrs)
        
        text = []
        # text.append(title)
        
        if len(action_dict) != 0:
            for key, value in action_dict.items():
                sent = key + ' ' +', '.join(ing for ing in value)
                text.append(sent)
        else:
            text.extend(ingrs)
        
        txt = '. '.join(text[:self.max_action])
        
        
        if self.initial == 'pretrain':
            
            title_encoding = self.tokenizer(
                title,
                truncation=True,
                # padding='max_length',
                # max_length=self.max_text_len,
                # return_special_tokens_mask=True,
                return_tensors='pt',
            )    
            
            txt_encoding = self.tokenizer(
                txt,
                truncation=True,
                # padding='max_length',
                # max_length=self.max_text_len,
                # return_special_tokens_mask=True,
                return_tensors='pt',
            )   
            
            title = title_encoding['input_ids'].squeeze(0)[:self.max_text_len]
            act_ing = txt_encoding['input_ids'].squeeze(0)[:self.max_text_len]
            
        else:
            title = torch.Tensor(get_token_ids(title, self.vocab)[:self.max_text_len])
            act_ing = torch.Tensor(get_token_ids(txt, self.vocab)[:self.max_text_len])
        
        return recipe_id, title, act_ing, img


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
                mode='train', augment=True, drop_last=True, initial='scratch'):

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

    ds = Recipe1M(root, transform=transforms_, split=split, initial=initial)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=multiprocessing.cpu_count(),
                        collate_fn=collate_fn, drop_last=drop_last)

    return loader, ds

