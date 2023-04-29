import torch
import pickle
import os
import random
random.seed(1234)
from PIL import Image
from torch.utils.data import Dataset
import nltk
# from vilt.transforms import keys_to_transforms
from transformers import ViTFeatureExtractor, ViTModel
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

invisible_ing = ['salt', 'sugar', 'vinegar', 'pepper', 'powder', 'butter', 'flour', 
                 'sauce', 'cumin', 'spice', 'oregano', 'rosemary','thyme',
                 'flakes', 'nutmeg', 'cayenne', 'cinnamon', 'cloves','garlic',
                 'ginger', 'paprika', 'rub', 'blend', 'hickory', 'hanout', 'extract'
                 'tamari', 'mesquite', 'seasoning']

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

class RetrievalDataset(Dataset):

    def __init__(self, tokenizer, args, split='train', is_train=True, fast_run=False):

        self.root = args.data_dir
        self.data = pickle.load(open(os.path.join(self.root,'traindata', split + '.pkl'), 'rb'))
        # self.transform = transform
        self.split = split
        self.tokenizer = tokenizer
        # self.transforms = keys_to_transforms(transforms, size=image_size)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch32-224-in21k")
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.args = args
        self.is_train=is_train
        self.max_ingrs = 20
        self.max_intrs = 20
        self.max_tags = 5
        self.max_sents = 15
        self.cross_image_eval = args.cross_image_eval
            
        if args.add_tag:
            self.val_ing = pickle.load(open(os.path.join(self.root,'valid_ingredients.pkl'),'rb'))
            with open(os.path.join(self.root, 'classes1M.pkl'),'rb') as f:
                self.class_dict = pickle.load(f)
                self.id2class = pickle.load(f)
            
        self.ids = list(self.data.keys())    
        self.index_mapper=dict()
        self.img_names = list()
            
        if self.split == 'train' and fast_run==False:
            j=0
            _j=0
            for i, id in enumerate(self.ids):
                img_list = self.data[id]['images']
                if len(img_list)<=5:
                    for img in img_list:
                        self.img_names.append(img)
                        self.index_mapper[j]= (i, _j)
                        j += 1
                        _j += 1
                else:
                    # random choose 5 images when # images > 5
                    # random_idx = random.sample([n for n in range(len(img_list))], 5)
                    random_choie = random.sample(img_list, 5)
                    for img in random_choie:
                        self.img_names.append(img)
                        self.index_mapper[j] = (i, _j)
                        j += 1
                        _j += 1
        elif self.split in ['val', 'test'] and fast_run==False:
            # random select 1000 subset from val/test set
            self.random_sample = random.sample(self.ids, 1000)
            for _, id in enumerate(self.random_sample):
                img_list = self.data[id]['images']
                # random choose one image
                # _j = random.choice([n for n in range(len(img_list))])
                random_choice = random.choice(img_list)
                self.img_names.append(random_choice)
            j = 0
            for i in range(len(self.random_sample)):
                # if not args.cross_image_eval:
                # # random choose 20 negative pairs
                #     self.samples = random.sample(self.img_names, 20)
                #     for img in self.samples:
                #         self.index_mapper[j] = (i, self.img_names.index(img))
                #         j += 1
                # else:
                for _j in range(len(self.img_names)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
                        
        elif self.split =='train' and fast_run ==True:
            j=0
            _j=0
            for i, id in enumerate(self.ids[:100]):
                img_list = self.data[id]['images']
                # if len(img_list)<=5:
                for img in img_list:
                    self.img_names.append(img)
                    self.index_mapper[j]= (i, _j)
                    j += 1
                    _j += 1
                    
        elif self.split in ['val', 'test'] and fast_run==True:
            self.random_sample = random.sample(self.ids, 100)
            for _, id in enumerate(self.random_sample):
                img_list = self.data[id]['images']
                random_choice = random.choice(img_list)
                self.img_names.append(random_choice)
            j = 0
            for i in range(len(self.random_sample)):
                for _j in range(len(self.img_names)):
                    self.index_mapper[j] = (i, _j)
                    j += 1


    def __len__(self):
        return len(self.index_mapper)

    def get_ids(self):
        return self.ids

    def get_image(self, img_idx):
        # _, img_idx = self.index_mapper[index]
        # entry = self.data[self.ids[rep_idx]]
        img_name = self.img_names[img_idx]
        img_name = '/'.join(img_name[:4])+'/'+img_name
        img = Image.open(os.path.join(self.root, self.split, img_name))
        return img

    def get_image_feat(self, img):
        # image_tensor = [tr(img) for tr in self.transforms]
        image = self.feature_extractor(img, return_tensors='pt')
        img_tensor = image['pixel_values']
        model = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k")
        
        with torch.no_grad():
            outputs = model(img_tensor)
        img_feat = outputs.last_hidden_state.squeeze(0)
        return img_feat
        
    def get_label(self, index):
        rep_idx, img_idx = self.index_mapper[index]
        
        if self.split=='train':
            rep_id = self.ids[rep_idx]
        else:
            rep_id = self.random_sample[rep_idx]
            
        img_list = self.data[rep_id]['images']
        return 1 if self.img_names[img_idx] in img_list else 0
        
    def get_od_labels(self, rep_idx):
        # rep_idx, _ = self.index_mapper[index]
        if self.split=='train':
            recipe_id = self.ids[rep_idx]
        else:
            recipe_id = self.random_sample[rep_idx]
            
        tags = []
        if self.args.add_tag:
            class_id = self.class_dict[recipe_id]
            if class_id != 0:
                rep_class = self.id2class[class_id]
                tags.append(rep_class)
            ing_list = self.val_ing[recipe_id]
            for ing in ing_list:
                words = nltk.tokenize.word_tokenize(ing)
                remove_list = []
                for word in words:
                    if word in invisible_ing:
                        remove_list.append(ing)  
                # remove invisible ingredients
                for inv in list(set(remove_list)):
                    ing_list.remove(inv)      
            tags.extend(ing_list)
        od_tags = ' '.join(tags[:self.max_tags])
        return od_tags
       
    def get_text(self, rep_idx):
        # rep_idx, _ = self.index_mapper[index]
        if self.split =='train':
            rep_id = self.ids[rep_idx]
            entry = self.data[rep_id]
        else:
            rep_id = self.random_sample[rep_idx]
            entry = self.data[rep_id]
        
        title = entry['title']
        ingrs = entry['ingredients'][:self.max_ingrs]
        # instrs = entry['instructions'][:self.max_intrs]
        action_ing = info_extract(rep_id, entry)

        text = []
        text.append(title)
        if len(action_ing) != 0:
            sents = [' '.join(sent) for sent in action_ing]
            text.extend(sents)
        else:
            text.extend(ingrs)
            
        txt = ' '.join(text[:self.max_sents])  
        return txt
        
    def tensorize_example(self, text_a, img_feat, text_b=None, 
                          cls_token_segment_id=0, pad_token_segment_id=0,
                          sequence_a_segment_id=0, sequence_b_segment_id=1):
        
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > (self.max_seq_len - 2 - 10):
            tokens_a = tokens_a[:(self.max_seq_len - 2 - 10)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        # att_mask_type = self.args.att_mask_type
        # if att_mask_type == "CLR":
        attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                        [1] * img_len + [0] * img_padding_len
        # else:
        #     # use 2D mask to represent the attention
        #     max_len = self.max_seq_len + self.max_img_seq_len
        #     attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        #     # full attention of C-C, L-L, R-R
        #     c_start, c_end = 0, seq_a_len
        #     l_start, l_end = seq_a_len, seq_len
        #     r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        #     attention_mask[c_start : c_end, c_start : c_end] = 1
        #     attention_mask[l_start : l_end, l_start : l_end] = 1
        #     attention_mask[r_start : r_end, r_start : r_end] = 1
        #     if att_mask_type == 'CL':
        #         attention_mask[c_start : c_end, l_start : l_end] = 1
        #         attention_mask[l_start : l_end, c_start : c_end] = 1
        #     elif att_mask_type == 'CR':
        #         attention_mask[c_start : c_end, r_start : r_end] = 1
        #         attention_mask[r_start : r_end, c_start : c_end] = 1
        #     elif att_mask_type == 'LR':
        #         attention_mask[l_start : l_end, r_start : r_end] = 1
        #         attention_mask[r_start : r_end, l_start : l_end] = 1
        #     else:
        #         raise ValueError("Unsupported attention mask type {}".format(att_mask_type))
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)
        
    def __getitem__(self, index):
        if self.is_train:
            rep_idx, img_idx = self.index_mapper[index]
            img = self.get_image(img_idx)
            feature = self.get_image_feat(img)
            caption = self.get_text(rep_idx)
            od_labels = self.get_od_labels(rep_idx)
            example = self.tensorize_example(caption, feature, text_b=od_labels)

            # select a negative pair
            neg_indexs = list(range(0, index)) + list(range(index + 1, len(self.index_mapper)))
            neg_rep_idx = rep_idx
            while neg_rep_idx == rep_idx:
                neg_idx = random.choice(neg_indexs)
                neg_rep_idx, neg_img_idx = self.index_mapper[neg_idx]
                
            if random.random() <= 0.5:
                caption_neg = self.get_text(neg_rep_idx)
                example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels)
            else:
                # randomly select a negative image 
                img_neg = self.get_image(neg_img_idx)
                feature_neg = self.get_image_feat(img_neg)
                od_labels_neg = self.get_od_labels(neg_rep_idx)
                example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg)

            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
            return index, example_pair
        else:
            rep_idx, img_idx = self.index_mapper[index]
            img = self.get_image(img_idx)
            feature = self.get_image_feat(img)
            caption = self.get_text(rep_idx)
            od_labels = self.get_od_labels(rep_idx)
            example = self.tensorize_example(caption, feature, text_b=od_labels)
            label = 1 if rep_idx == index else 0
            return index, tuple(list(example) + [label])