
import torch 
import pickle
import os
import random
import nltk
from PIL import Image
from torch.utils.data import Dataset
from module import info_extraction
# from module.utils import get_token_ids

invisible_ing = ['salt', 'sugar', 'vinegar', 'pepper', 'powder', 'butter', 'flour', 
                 'sauce', 'cumin', 'spice', 'oregano', 'rosemary','thyme', 'oil',
                 'flakes', 'nutmeg', 'cayenne', 'cinnamon', 'cloves','garlic',
                 'ginger', 'paprika', 'rub', 'blend', 'hickory', 'hanout', 'extract',
                 'tamari', 'mesquite', 'seasoning']


class Recipe1M(Dataset):
    
    def __init__(self, args, tokenizer, transforms=None, split='train', sample=1000):
        
        self.root = args.data_dir 
        self.data = pickle.load(open(os.path.join(self.root,'traindata', split + '.pkl'), 'rb'))
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.split = split
        self.max_seq_len = args.max_seq_len
        self.max_ingrs = 20
        self.max_instrs = 20
        self.max_tags = 6
        self.sample = sample
        self.extract_info = args.extract_info
        self.add_tags = args.add_tags
        self.do_fine_tune = args.do_fine_tune
        if split != 'train':
            self.draw_false_text = args.draw_false_text
        else:
            self.draw_false_text = 0
        
        # if self.tokenizer == None:
        #     self.vocab_inv = pickle.load(open(os.path.join(self.root,'vocab.pkl'),'rb'))
        #     self.vocab = {}
        #     for k, v in self.vocab_inv.items():
        #         if type(v) != str:
        #             v = v[0]
        #         self.vocab[v] = k
                
        if self.add_tags:
            self.val_ing = pickle.load(open(os.path.join(self.root,'valid_ingredients.pkl'),'rb'))
            with open(os.path.join(self.root, 'classes1M.pkl'),'rb') as f:
                self.class_dict = pickle.load(f)
                self.id2class = pickle.load(f)
        
        self.ids = list(self.data.keys())
        
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
            if sample == 'all':
                for i, id in enumerate(self.ids):
                    img_list = self.data[id]['images']
                    # random choose 1 image from img_list
                    img = random.choice(img_list)
                    self.index_mapper[i] = (id, img)
            else:
                # randomly select sample (default 1000) from val or test dataset
                random_sample = random.sample(self.ids, sample)
                for i, id in enumerate(random_sample):
                    img_list = self.data[id]['images']
                    # random choose 1 image from img_list
                    img = random.choice(img_list)
                    self.index_mapper[i] = (id, img)
                
 
    def __len__(self):
        return len(self.index_mapper)

    def get_ids(self):
        return self.ids
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size
  
        
    def get_image(self, img_name):
        img_path = '/'.join(img_name[:4])+'/'+ img_name
        img = Image.open(os.path.join(self.root, self.split, img_path))
        if self.transforms != None:
            img = [self.transforms(img)]
            
        return {'img_name': img_name,
                'image': img}
       
    
    def get_object_tags(self, rep_id):
         
        tags = []
        class_id = self.class_dict[rep_id]
        if class_id != 0:
            rep_class = self.id2class[class_id]
            tags.append(rep_class)
        ing_list = self.val_ing[rep_id].copy()
        remove_list = []
        for ing in ing_list:
            words = nltk.tokenize.word_tokenize(ing)
            for word in words:
                if word in invisible_ing:
                    remove_list.append(ing)  
        # remove invisible ingredients
        try:
            for inv in remove_list:
                ing_list.remove(inv)      
        except ValueError:
            pass
        tags.extend(list(set(ing_list)))
        object_tags = ' '.join(tags[:self.max_tags])
        
        return object_tags   
       
    def get_text(self, rep_id):
        
        entry = self.data[rep_id]
        
        title = entry['title']
        ingredients = entry['ingredients'][:self.max_ingrs]
        instructions = entry['instructions'][:self.max_instrs]
        
        if self.extract_info:
            act_ing = info_extraction.info_extract(rep_id, entry)
            txt = [' '.join(sent) for sent in act_ing]
            act_ing_txt = '. '.join(txt)
            text = title + '. ' + act_ing_txt
        else:
            ing_txt = '. '.join(ingredients)
            ins_txt = ' '.join(instructions)
            text = title + '. ' + ing_txt + ins_txt
        
        return text 
        
    def tensorize(self, text):
        # if self.tokenizer:
        encoding = self.tokenizer(text, 
                                    padding='max_length',
                                    truncation = True,
                                    max_length = self.max_seq_len)
        return encoding
    
    def get_false_text(self, rep_id, i):
        ids_copy = self.ids.copy()
        ids_copy.remove(rep_id)
        neg_rep_id = random.choice(ids_copy)
        
        neg_txt = self.get_text(neg_rep_id)
        encoding = self.tensorize(neg_txt)
        
        return {f"false_text_{i}": (neg_txt, encoding)}
    
    def __getitem__(self, index):
        rep_id, img_name = self.index_mapper[index]
        ret = dict()
        ret['raw_index'] = index
        
        img = self.get_image(img_name)
        ret.update(img)
        
        text = self.get_text(rep_id)
        
        ids_copy = self.ids.copy()
        ids_copy.remove(rep_id)
        neg_rep_id = random.choice(ids_copy)     
        
        label = None
        if self.extract_info and self.add_tags and not self.do_fine_tune:
            if self.split =='train' and random.random() <= 0.5:
                neg_object_tags = self.get_object_tags(neg_rep_id)
                text = text + '. ' + neg_object_tags
                label = 0
            else:
                object_tags = self.get_object_tags(rep_id)
                text = text + '. ' + object_tags
                label = 1
        ret.update({'label': label})
        
        encoding = self.tensorize(text)  
        txt = {'rep_id': rep_id,
                'text': (text, encoding)} 
        ret.update(txt)
        
        for i in range(self.draw_false_text):
            ret.update(self.get_false_text(rep_id,i))
        
        neg_img_name = random.choice(self.data[neg_rep_id]['images'])
        neg_image = self.get_image(neg_img_name)
        neg_img =  {'neg_rep_id': neg_rep_id,
                'neg_img_name': neg_img_name,
                'neg_image': neg_image['image']}
        
        ret.update(neg_img)
        return ret
        

def collate(batch, mlm_collator):
        
    batch_size = len(batch)
    keys = set([k for b in batch for k in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    
    img_keys = [k for k in list(dict_batch.keys()) if "image" in k]

    for img_key in img_keys:
        img = dict_batch[img_key]
        view_size = len(img[0])

        new_images = [
            torch.zeros(batch_size, 3, 224, 224)
            for _ in range(view_size)
        ]

        for bi in range(batch_size):
            orig_batch = img[bi]
            for vi in range(view_size):
                if orig_batch is None:
                    new_images[vi][bi] = None
                else:
                    orig = img[bi][vi]
                    new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

        dict_batch[img_key] = new_images
    
    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

    if len(txt_keys) != 0:
        texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        # draw_text_len = len(encodings)
        flatten_encodings = [e for encoding in encodings for e in encoding]
        # print(flatten_encodings)
        flatten_mlms = mlm_collator(flatten_encodings)

        for i, txt_key in enumerate(txt_keys):
            texts, encodings = (
                [d[0] for d in dict_batch[txt_key]],
                [d[1] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = torch.zeros_like(mlm_ids)
            attention_mask = torch.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                _input_ids, _attention_mask = (
                    torch.tensor(encoding["input_ids"]),
                    torch.tensor(encoding["attention_mask"]),
                )
                input_ids[_i, : len(_input_ids)] = _input_ids
                attention_mask[_i, : len(_attention_mask)] = _attention_mask

            dict_batch[txt_key] = texts
            dict_batch[f"{txt_key}_ids"] = input_ids
            dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
            dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            dict_batch[f"{txt_key}_masks"] = attention_mask

    return dict_batch

                             