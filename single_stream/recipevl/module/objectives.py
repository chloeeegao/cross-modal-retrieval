import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import functools
import tqdm 
from transformers import DataCollatorForLanguageModeling
from dataset import collate
from dist_utils import all_gather

def accuracy(logits, target):
    
    correct = 0
    total = 0
    
    logits, target = (
        logits.detach(),
        target.detach(),
    )
    preds = logits.argmax(dim=-1)
    preds = preds[target != -100]
    target = target[target != -100]
    if target.numel() == 0: # return number of elements in the tensor
        return 1

    assert preds.shape == target.shape

    correct += torch.sum(preds == target)
    total += target.numel()

    return correct / total


def compute_mlm(model, batch):
    device = dist.get_rank()
    module = model.module if hasattr(model, 'module') else model
    
    infer = module.infer(batch, device, mask_text=True, mask_image=False)
    mlm_logits = module.mlm_score(infer["text_feats"].to(device))
    mlm_labels = infer["text_labels"].to(device)

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, module.args['vocab_size']),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }
    
    acc = accuracy(ret["mlm_logits"], ret["mlm_labels"])

    ret['mlm_accuracy'] = acc

    return ret

def compute_itm(model, batch):
    
    device = dist.get_rank()
    module = model.module if hasattr(model, 'module') else model
    
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(device)
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["neg_image"])
    ]

    batch["image"] = itm_images

    infer = module.infer(batch, device, mask_text=False, mask_image=False)
    itm_logits = module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    acc = accuracy(ret["itm_logits"], ret["itm_labels"])

    ret['itm_accuracy'] = acc
    
    return ret


@torch.no_grad()
def compute_irtr_recall(model, dataset, tokenizer):
    
    device = dist.get_rank()
    text_dset = dataset
    module = model.module if hasattr(model, 'module') else model
    
    mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability= 0.15
        )
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=16,
        pin_memory=True,
        collate_fn=functools.partial(
            collate,
            mlm_collator=mlm_collator,
        ),
        drop_last=False
    )

    image_dset = text_dset
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers= 16,
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            collate,
            mlm_collator=mlm_collator,
        ),
    )
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(device),
                "text_masks": _b["text_masks"].to(device),
                "text_labels": _b["text_labels"].to(device),
                "raw_index": _b["raw_index"],
            }
        )
    tiids = list()
    for pre in text_preload:
        tiids += pre["raw_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im) = module.transformer.visual_embed(
            _b["image"][0].to(device),
        )
        image_preload.append((ie, im, _b["raw_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                score = module.rank_output(
                        module.infer(
                            {
                                "text_ids": txt_batch["text_ids"],
                                "text_masks": txt_batch["text_masks"],
                                "text_labels": txt_batch["text_labels"],
                            },
                            device,
                            image_embeds=ie,
                            image_masks=im,
                        )["cls_feats"]
                    )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

