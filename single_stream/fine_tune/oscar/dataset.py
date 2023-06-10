import torch
import pickle
import os
import random
from torch.utils.data import Dataset
import nltk
import spacy
from spacy.pipeline import EntityRuler
import re
import copy
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTModel

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
                 'sauce', 'cumin', 'spice', 'oregano', 'rosemary','thyme', 'oil',
                 'flakes', 'nutmeg', 'cayenne', 'cinnamon', 'cloves','garlic',
                 'ginger', 'paprika', 'rub', 'blend', 'hickory', 'hanout', 'extract',
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

    def __init__(self, args, tokenizer, split='train', is_train=True):

        self.root = args.data_dir
        self.data = pickle.load(open(os.path.join(self.root,'traindata', split + '.pkl'), 'rb'))
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.max_tags = 6
        self.args = args
        self.is_train=is_train
        self.fast_run = args.fast_run
            
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch32-224-in21k")    
        
        if args.add_tag:
            self.val_ing = pickle.load(open(os.path.join(self.root,'valid_ingredients.pkl'),'rb'))
            with open(os.path.join(self.root, 'classes1M.pkl'),'rb') as f:
                self.class_dict = pickle.load(f)
                self.id2class = pickle.load(f)
          
        self.ids = list(self.data.keys())
            
        if split =='train' and is_train and self.fast_run:
            self.run_ids = random.sample(self.ids , 1000)
        elif split in ['val', 'test'] and self.fast_run:
            self.run_ids = random.sample(self.ids, 60)
        elif split =='train' and is_train and not self.fast_run:
            # self.run_ids = random.sample(self.ids, 100000)
            self.run_ids = self.ids
        elif split in ['val', 'test'] and not self.fast_run:
            self.run_ids = random.sample(self.ids, 1000)
        else:
            self.run_ids = self.ids

    def __len__(self):
        return len(self.run_ids)

    def get_image_feat(self, rep_id):
        transforms_list = [transforms.Resize((256))]
        # if self.split == 'train':
        #     # Image preprocessing, normalization 
        #     transforms_list.append(transforms.RandomHorizontalFlip())
        #     transforms_list.append(transforms.RandomCrop(224))
        # else:
        transforms_list.append(transforms.CenterCrop(224))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                    (0.229, 0.224, 0.225)))
        transforms_ = transforms.Compose(transforms_list)
        # random choice a image
        img_name = random.choice(self.data[rep_id]['images'])
        img_path = '/'.join(img_name[:4])+'/'+ img_name
        img = Image.open(os.path.join('/data/s2478846/data', self.split, img_path)) 
        img_tensor = transforms_(img).unsqueeze(0)  
        with torch.no_grad():
            outputs = self.vit_model(img_tensor)
            img_feat = outputs.last_hidden_state.squeeze(0)

        return img_feat
         
    def get_object_tags(self, rep_id):
         
        tags = []
        if self.args.add_tag:
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
        ingrs = entry['ingredients']
        # instrs = entry['instructions'][:self.max_intrs]
        action_ing = info_extract(rep_id, entry)

        text = []
        text.append(title)
        if len(action_ing) != 0:
            sents = [' '.join(sent) for sent in action_ing]
            text.extend(sents)
        else:
            text.extend(ingrs)
            
        txt = ' '.join(text)  
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
        attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                        [1] * img_len + [0] * img_padding_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)
        
    def __getitem__(self, index):
        if self.is_train:
            rep_id = self.run_ids[index]
            # img = self.get_image(img_idx)
            img_feature = self.get_image_feat(rep_id)
            recipe_text = self.get_text(rep_id)
            object_tags = self.get_object_tags(rep_id)
            example = self.tensorize_example(recipe_text, img_feature, text_b=object_tags)

            # select a negative pair
            ids_copy = self.run_ids.copy()
            ids_copy.remove(rep_id)
            neg_rep_id = random.choice(ids_copy)
                
            if random.random() <= 0.5:
                neg_recipe = self.get_text(neg_rep_id)
                example_neg = self.tensorize_example(neg_recipe, img_feature, text_b=object_tags)
            else:
                # randomly select a negative image 
                img_neg = self.get_image_feat(neg_rep_id)
                object_tags_neg = self.get_object_tags(neg_rep_id)
                example_neg = self.tensorize_example(recipe_text, img_neg, text_b=object_tags_neg)

            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
            return index, example_pair
        else:
            rep_id = self.run_ids[index]
            img_feat = self.get_image_feat(rep_id)
            recipe_text = self.get_text(rep_id)
            object_tags = self.get_object_tags(rep_id)
            example = self.tensorize_example(recipe_text, img_feat, text_b=object_tags)
            label = 1
            return index, tuple(list(example) + [label])