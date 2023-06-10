import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DataCollatorForLanguageModeling
from torchvision import transforms
import functools
import tqdm
import random
import pickle
from collections import defaultdict
from module.model import RecipeVL
from dataset import Recipe1M, collate
from config import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case='uncased')

def get_dataset(args, n_sample):
    
    transforms_list = [transforms.Resize((256))]
    transforms_list.append(transforms.CenterCrop(224))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))
    transforms_ = transforms.Compose(transforms_list)
    test_dataset = Recipe1M(args, tokenizer, transforms=transforms_, split='test', sample=n_sample)
  
    return test_dataset


def load_model(args, weights_path):
    
    loss_names = {'itm': 1, 'mlm': 1, 'irtr': 1}
    model = RecipeVL(args, loss_names)
    checkpoint = torch.load(os.path.join(weights_path, 'model_optim.pt'), map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']
    # opt_state_dict = checkpoint['optim_state_dict']
    if hasattr(model, "module"):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    return model

def get_topk(dataset, index, model, device, batch_size, mode='im2rec'):
    
    recipe_id = dataset[index]['rep_id']
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability= 0.15) 
    
    model = model.to(device)
    module = model.module if hasattr(model, 'module') else model
    
    if mode == 'im2rec':
        image_dset = dataset[index]
        text_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            collate_fn=functools.partial(
                collate,
                mlm_collator=mlm_collator,
            ),
            drop_last=False
        )
        
        text_preload = list()
        for _, batch in enumerate(text_loader):
            text_preload.append(
                {
                    "text_ids": batch["text_ids"],
                    "text_masks": batch["text_masks"],
                    "text_labels": batch["text_labels"],
                    "raw_index": batch["raw_index"],
                }
            )

        image_preload = list()
        (ie, im) = module.transformer.visual_embed(
            image_dset['image'][0].unsqueeze(0).to(device),
        )
        image_preload.append((ie, im, image_dset["raw_index"]))
        
    else:
        text_dset = [dataset[index]]
        txt_batch = collate(text_dset, mlm_collator)
   
        image_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers= 16,
            pin_memory=True,
            collate_fn=functools.partial(
                collate,
                mlm_collator=mlm_collator,
            ),
        )
        
        text_preload = list()
        text_preload.append(
            {
                "text_ids": txt_batch["text_ids"],
                "text_masks": txt_batch["text_masks"],
                "text_labels": txt_batch["text_labels"],
                "raw_index": txt_batch["raw_index"],
            }
        )

        image_preload = list()
        for _, batch in enumerate(image_loader):
            (ie, im) = module.transformer.visual_embed(
                batch['image'][0].to(device),
            )
            image_preload.append((ie, im, batch["raw_index"]))
            torch.cuda.empty_cache()
            
    rank_scores = list()
    rank_iids = list()        

    if mode =='im2rec':
        _, l, c = ie.shape
        ie = ie.expand(batch_size, l, c)
        im = im.expand(batch_size, l)
        for _b in text_preload:
            _iid = _b['raw_index']
            batch_score = list()    
            with torch.cuda.amp.autocast():
                score = module.rank_output(
                        module.infer(
                            {
                                "text_ids": _b["text_ids"],
                                "text_masks": _b["text_masks"],
                                "text_labels": _b["text_labels"],
                            },
                            device=device,
                            image_embeds=ie,
                            image_masks=im,
                        )["cls_feats"]
                    )[:, 0]
                score = score.detach()
                torch.cuda.empty_cache()
            batch_score.append(score)
            batch_score = torch.cat(batch_score)
            rank_scores.append(batch_score.tolist())
            rank_iids.append(_iid)
        
    else:
        l, c = txt_batch['text_ids'].shape
        txt_batch["text_ids"] = txt_batch["text_ids"].expand(batch_size, c)
        txt_batch["text_masks"] = txt_batch["text_masks"].expand(batch_size, c)
        txt_batch["text_labels"] = txt_batch["text_labels"].expand(batch_size, c)
        for _b in image_preload:
            ie, im, _iid = _b
            batch_score = list()    
            with torch.cuda.amp.autocast():
                score = module.rank_output(
                        module.infer(
                            {
                                "text_ids": txt_batch["text_ids"],
                                "text_masks": txt_batch["text_masks"],
                                "text_labels": txt_batch["text_labels"],
                            },
                            device=device,
                            image_embeds=ie,
                            image_masks=im,
                        )["cls_feats"]
                    )[:, 0]
                score = score.detach()
                torch.cuda.empty_cache()
            batch_score.append(score)
            batch_score = torch.cat(batch_score)
            rank_scores.append(batch_score.tolist())
            rank_iids.append(_iid)
    
    iids = torch.tensor(rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=0)
    topk10_iids = iids[topk10.indices]
    top_10 = [dataset[ind[0]]['rep_id'] for ind in topk10_iids.tolist()]
        
    return recipe_id, top_10


def main(args, device, batch_size, n_sample, load_path):
    dataset = get_dataset(args, n_sample=n_sample)
    model = load_model(args, load_path)
    im2rec_res = defaultdict(list)
    rec2im_res = defaultdict(list)
    index = random.sample([i for i in range(n_sample)],100)
    for i in tqdm.tqdm(index, desc='im2rec rank loop'):
        recipe_id, top_10 = get_topk(dataset, i, model, device, batch_size, mode='im2rec') 
        im2rec_res[recipe_id] = top_10
    pickle.dump(im2rec_res, open('im2rec_res.pkl', 'wb'))
    
    for i in tqdm.tqdm(index, desc='rec2im rank loop'):
        recipe_id, top_10 = get_topk(dataset, i, model, device, batch_size, mode='rec2im') 
        rec2im_res[recipe_id] = top_10
    pickle.dump(im2rec_res, open('rec2im_res.pkl', 'wb'))
    
    
if __name__ == '__main__':
    args = get_args()
    args.vit = 'vit_base_patch16_224'
    batch_size = 20
    n_sample = 1000
    load_path = 'output/vit_base_v8/checkpoint-7-89453'
    main(args, device, batch_size, n_sample, load_path)  
    # print('Recipe id: {}, Retrieved topk: {}'.format(recipe_id, top_10))