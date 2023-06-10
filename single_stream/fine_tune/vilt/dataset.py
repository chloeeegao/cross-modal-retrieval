import pickle
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from vilt.transforms import keys_to_transforms

import spacy 
from spacy.pipeline import EntityRuler
import re

nlp= spacy.load("/data/s2478846/cross_modal_retrieval/preprocessing/food_entity_extractor")
if 'entity_ruler' in nlp.pipe_names:
    nlp.remove_pipe("entity_ruler") 
ruler = EntityRuler(nlp)
nlp.add_pipe(ruler)
ruler.add_patterns([{"label": "ING", "pattern": "rice"}])
ruler.add_patterns([{"label": "ING", "pattern": "mixture"}])
ruler.add_patterns([{"label": "ING", "pattern": "meat"}])
ruler.add_patterns([{"label": "ING", "pattern": "chicken"}])
ruler.add_patterns([{"label": "ING", "pattern": "pork"}])
ruler.add_patterns([{"label": "ING", "pattern": "beef"}])
ruler.add_patterns([{"label": "ING", "pattern": "ingredients"}])
valid_ingredient = pickle.load(open('/data/s2478846/data/valid_ingredients.pkl','rb'))

cooking_verbs = ['add','bake', 'barbecue', 'baste', 'beat', 'blanch', 'blend', 'bring',
                 'boil', 'braise', 'bread', 'break','broil', 'brown','brush', 'candy', 'can', 
                 'carve', 'char', 'check', 'chill', 'chop', 'clean', 'coat', 
                 'combine', 'cook', 'cool', 'core', 'cover','cream', 'crisp', 'crush', 
                 'cube', 'cut', 'debone', 'decorate' 'deep-fry', 'dehydrate', 'dice', 'dip' 'dissolve', 
                 'drain', 'dress', 'drizzle', 'dry', 'drop','dust', 'emulsify', 'ferment', 'filet', 
                 'flame', 'flambé', 'flip', 'fold', 'freeze', 'fry', 'garnish', 'glaze', 'grate', 
                 'grill', 'grind', 'gut', 'heat', 'hull', 'infuse', 'julienne', 'knead', 'layer', 'level', 
                 'liquefy', 'light' 'marinate', 'mash', 'measure','melt', 'mince', 'mix', 'mold', 'move',
                 'microwave','oil', 'pack','pan-fry', 'parboil', 'pare', 'peel', 'place', 'pickle', 'pierce', 'pinch', 'pit', 'poach', 'pop', 
                 'pour', 'preheat', 'preserve', 'pressure-cook', 'prick', 'puree', 'push', 'put', 'reduce', 'remove', 'rinse',
                 'refrigerate', 'roast', 'roll', 'sauté', 'saute','serve', 'scald', 'scramble','scallop', 'score', 'sear', 
                 'season', 'shred', 'simmer', 'sip','sift', 'skewer', 'slice', 'smoke', 'smooth', 'soak', 'soften', 'sprinkle', 
                 'sous-vide', 'spatchcock', 'spice', 'spread', 'squeeze','steam', 'steep', 'stir', 'strain', 'stick',
                 'stuff', 'submerge', 'sweeten', 'swirl', 'taste', 'take','temper', 'tenderize', 'thicken', 'toast', 'top',
                 'toss', 'truss', 'thread', 'turn on', 'turn off', 'wash','weight','whip', 'whisk' 'wilt']

def info_extract(recipe_id, recipe):
    
    ingredients = recipe['ingredients']
    instructions = recipe['instructions']

    entity_list = valid_ingredient[recipe_id]
    for sent in ingredients:
        sent = re.sub(r'[^a-zA-Z ]', ' ', sent)
        doc = nlp(sent)
        for token in doc:
            if token.dep_ =='ROOT' and token.pos_ == 'NOUN':
                entity_list.append(token.text)
            elif token.dep_ in ['dobj','nsubj','pobj'] and token.pos_ =='NOUN':
                entity_list.append(token.text)
    entity_list = list(set(entity_list))      
    patterns = []
    for ent in entity_list:
        patterns.append({"label": "ING", "pattern": ent})
    ruler.add_patterns(patterns)
        
    action_ing = []
    for sent in instructions:
        doc = nlp(sent)
        tmp = []
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ING','FOOD']]
        if 'ingredients' in entities:
            entities = entity_list
        for token in doc:
            if token.dep_ == 'ROOT' and token.lemma_ in cooking_verbs:
                if len(tmp) == 0:
                    tmp.append(token.text)
                    tmp.extend(entities)
            elif token.dep_ in ['compound', 'nmod', 'xcomp', 'amod'] and token.lemma_ in cooking_verbs:
                if len(tmp) == 0:
                    tmp.append(token.text)
                    tmp.extend(entities)
        if len(tmp) != 0:
            action_ing.append(tmp)

    return action_ing


class Recipe1MDataset(Dataset):
    def __init__(self, root, 
                 transforms,
                 image_size,
                 split,
                 max_text_len=40,
                 draw_false_image=0,
                 draw_false_text=0,
                 image_only=False,
                 ):
        super(Recipe1MDataset).__init__()
        
        self.data = pickle.load(open(os.path.join(root,'traindata', split + '.pkl'), 'rb'))
        self.root = root
        self.split = split
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text 
        self.max_text_len = max_text_len
        self.max_ingrs = 20
        self.max_instrs = 20
        self.image_only = image_only
        
        self.transforms = keys_to_transforms(transforms, size=image_size)
 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case='uncased')

        self.ids = list(self.data.keys())
        self.index_mapper = dict()
        
        if self.split=='train' and not self.image_only:
            # self.sample = random.sample(self.ids, 100000)
            self.sample = self.ids
        else:
            # random select 1000 subset from val/test set
            self.sample = random.sample(self.ids, 1000)
            
        for i, id in enumerate(self.sample):
            # random choice one image
            img = random.choice(self.data[id]['images'])
            self.index_mapper[i] = (id, img)
                
    def __len__(self):
        return len(self.index_mapper)
    
    
    def get_raw_image(self, img_name):
        
        img_name = '/'.join(img_name[:4])+'/'+img_name
        img = Image.open(os.path.join(self.root, self.split, img_name))

        return img
    
    def get_image(self, img_name):
        image = self.get_raw_image(img_name)
        image_tensor = [tr(image) for tr in self.transforms]
        
        return {
            "img_name": img_name,
            "image": image_tensor,
            # "img_index": self.index_mapper[index][1],
            # "id_index": self.index_mapper[index][0],
            # "raw_index": index,
        }
        
    def get_false_image(self, rep_id, i):
        ids_copy = self.sample.copy()
        ids_copy.remove(rep_id)
        neg_rep_id = random.choice(ids_copy)
        entry = self.data[neg_rep_id]
        img_name = random.choice(entry['images'])
        # random_index = random.randint(0, len(self.index_mapper) - 1)
        
        image = self.get_raw_image(img_name)
        image_tensor = [tr(image) for tr in self.transforms]
        
        return {f"false_image_{i}": image_tensor}


    def get_text(self, rep_id):
        
        entry = self.data[rep_id]
        
        title = entry['title']
        ingrs = entry['ingredients'][:self.max_ingrs]
        # instrs = entry['instructions'][:self.max_instrs]
        
        action_ing = info_extract(rep_id, entry)
        
        text = []
        text.append(title)
        
        if len(action_ing) != 0:
            sents = [' '.join(sent) for sent in action_ing]
            text.extend(sents)
        else:
            text.extend(ingrs)
            
        txt = ' '.join(text)      

        encoding = self.tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "rep_id": rep_id,
            "text": (text, encoding),
            # "img_index": img_index,
            # "id_index": id_index,
            # "raw_index": index,
        }

    def get_false_text(self, rep_id, i):
        ids_copy = self.sample.copy()
        ids_copy.remove(rep_id)

        neg_rep_id = random.choice(ids_copy)
        entry = self.data[neg_rep_id]
            
        title = entry['title']
        ingrs = entry['ingredients'][:self.max_ingrs]
        # instrs = entry['instructions'][:self.max_instrs]
        
        action_ing = info_extract(rep_id, entry)
        
        text = []
        text.append(title)
        
        if len(action_ing) != 0:
            sents = [' '.join(sent) for sent in action_ing]
            text.extend(sents)
        else:
            text.extend(ingrs)
            
        txt = ' '.join(text)      

        encoding = self.tokenizer(
            txt,
            padding = 'max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{i}": (text, encoding)}

    def __getitem__(self, index):
        result = None
        rep_id, img_name = self.index_mapper[index]
        while result is None:
            try:
                ret = dict()
                ret['raw_index'] = index
                ret.update(self.get_image(img_name))
                # print('image ok')
                if not self.image_only:
                    txt = self.get_text(rep_id)
                    # print('text ok')
                    # ret.update({"replica": True if txt["img_index"] > 0 else False})
                    ret.update(txt)
                
                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(rep_id, i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(rep_id, i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in dataset -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
                
        return ret


    def collate(self, batch, mlm_collator):
        
        batch_size = len(batch)
        keys = set([k for b in batch for k in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
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
    