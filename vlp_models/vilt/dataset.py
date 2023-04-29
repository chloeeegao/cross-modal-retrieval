import pickle
import os
import random
random.seed(1234)
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from vilt.transforms import keys_to_transforms


import spacy 


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
        self.max_action = 8
        self.image_only = image_only
        
        self.transforms = keys_to_transforms(transforms, size=image_size)
 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case='uncased')

        self.ids = list(self.data.keys())
        self.index_mapper = dict()
        self.img_names = list()
        
        if self.split=='train' and not self.image_only:
            j=0
            for i, id in enumerate(self.ids):
                img_list = self.data[id]['images']
                if len(img_list)<=5:
                    for _j in range(len(img_list)):
                        self.img_names.append(img_list[_j])
                        self.index_mapper[j]= (i, _j)
                        j += 1
                else:
                    # random choose 5 images when # images > 5
                    # random_idx = random.sample([n for n in range(len(img_list))], 5)
                    random_choie = random.sample(img_list, 5)
                    for img in random_choie:
                        self.img_names.append(img)
                        self.index_mapper[j] = (i, img_list.index(img))
                        j += 1
        else:
            # random select 1000 subset from val/test set
            j = 0
            self.random_sample = random.sample(self.ids, 1000)
            for i, id in enumerate(self.random_sample):
                img_list = self.data[id]['images']
                # random choice one image
                random_choice = random.choice(img_list)
                self.img_names.append(random_choice)
                self.index_mapper[j] = (i, img_list.index(random_choice))
                j += 1
                    

    
        #     j=0
        #     for i, id in enumerate(self.ids):
        #         img_names = self.data[id]['images']
        #         for _j in range(len(img_names)):
        #             self.index_mapper[j]= (i, _j)
        #             j += 1
        # elif self.one_image and not self.eval:
        #     for i in range(len(self.data)):
        #         img_names = self.data[self.ids[i]]['images']
        #         j = random.randint(0, len(img_names))
        #         self.index_mapper[i] = (i, j)

        # elif self.eval and self.one_image:
        #     idx = random.sample(range(0, len(self.ids)), k=self.k)
        #     self.sub_ids = np.array(self.ids)[idx]
        #     for i, id in enumerate(self.sub_ids):
        #         self.index_mapper[i] = (i, 0)
    
    def __len__(self):
        return len(self.index_mapper)
    
    
    def get_raw_image(self, index):
        _, img_index = self.index_mapper[index]
        
        # if self.split =='train':
        #     entry = self.data[self.ids[id_index]]
        # else:
        #     entry = self.data[self.random_sample[id_index]]

        img_name = self.img_names[img_index]
        
        img_name = '/'.join(img_name[:4])+'/'+img_name
        img = Image.open(os.path.join(self.root, self.split, img_name))

        return img
    
    def get_image(self, index):
        image = self.get_raw_image(index)
        image_tensor = [tr(image) for tr in self.transforms]
        
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][1],
            "id_index": self.index_mapper[index][0],
            "raw_index": index,
        }
        
    def get_false_image(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}


    def get_text(self, index):
        
        id_index, img_index = self.index_mapper[index]
        
        if self.split !='train':
            entry = self.data[self.random_sample[id_index]]
        else:
            entry = self.data[self.ids[id_index]]
        
        title = entry['title']
        ingrs = entry['ingredients'][:self.max_ingrs]
        instrs = entry['instructions'][:self.max_instrs]
        
        action_dict = info_extract(instrs)
        
        # if len(action_dict) == 0:
        #     action_dict = info_extract(ingrs)
        
        text = []
        text.append(title)
        
        if len(action_dict) != 0:
            for key, value in action_dict.items():
                sent = key + ' ' +', '.join(ing for ing in value)
                text.append(sent)
        else:
            text.extend(ingrs)
        
        txt = '. '.join(text[:self.max_action])
        encoding = self.tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text": (text, encoding),
            "img_index": img_index,
            "id_index": id_index,
            "raw_index": index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        id_index, _ = self.index_mapper[random_index]
        
        if self.split !='train':
            entry = self.data[self.random_sample[id_index]]
        else:
            entry = self.data[self.ids[id_index]]
            
        
        title = entry['title']
        ingrs = entry['ingredients'][:self.max_ingrs]
        instrs = entry['instructions'][:self.max_instrs]
        
        
        action_dict = info_extract(instrs)
        
        # if len(action_dict) == 0:
        #     action_dict = info_extract(ingrs)
        
        text = []
        text.append(title)
        
        if len(action_dict) != 0:
            for key, value in action_dict.items():
                sent = key + ' ' +', '.join(ing for ing in value)
                text.append(sent)
        else:
            text.extend(ingrs)
        
        txt = '. '.join(text[:self.max_action])

        encoding = self.tokenizer(
            txt,
            padding = 'max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def __getitem__(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                # print('image ok')
                if not self.image_only:
                    txt = self.get_text(index)
                    # print('text ok')
                    ret.update({"replica": True if txt["img_index"] > 0 else False})
                    ret.update(txt)
                
                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
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
    