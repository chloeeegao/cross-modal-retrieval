
import os
import os.path as op
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from utils.logger import setup_logger
from utils.misc import mkdir, set_seed
from modeling.modeling_bert import ImageBertForSequenceClassification
from pytorch_transformers import BertTokenizer, BertConfig 
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.optim import AdamW

from config import get_args
from dataset import RetrievalDataset

args = get_args()
global logger
mkdir(args.output_dir)
logger = setup_logger("vlpretrain", args.output_dir, 0)

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

def compute_score_with_logits(logits, labels):
    if logits.shape[1] > 1:
        logits = torch.max(logits, 1)[1].data # argmax
        scores = logits == labels 
    else:
        scores = torch.zeros_like(labels).cuda()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores

def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


def train(args, train_dataset, val_dataset, model, tokenizer, rank=None):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if rank != None:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs \
                // args.gradient_accumulation_steps
                
    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))
        
    if rank != None:
        logger.info("*****Using DDP**********")
        model = model.to(rank)
        # wrap with DDP, this step will synch model across all the processes
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    elif args.n_gpu > 1 and rank == None:
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

    if rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc =0,  0.0, 0.0
    log_json = []
    best_score = 0
    count_step = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, (_, batch) in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(rank) for t in batch)
            inputs = {
                'input_ids':      torch.cat((batch[0], batch[5]), dim=0),
                'attention_mask': torch.cat((batch[1], batch[6]), dim=0),
                'token_type_ids': torch.cat((batch[2], batch[7]), dim=0),
                'img_feats':      torch.cat((batch[3], batch[8]), dim=0),
                'labels':         torch.cat((batch[4], batch[9]), dim=0)
            }
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            # when using DDP no need to do this
                # if args.n_gpu > 1: 
                #     loss = loss.mean() # mean() to average on multi-gpu parallel training
                # if args.gradient_accumulation_steps > 1:
                #     loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            batch_score = compute_score_with_logits(logits, inputs['labels']).sum()
            batch_acc = batch_score.item() / (args.train_batch_size * 2)
            global_loss += loss.item()
            global_acc += batch_acc
            count_step +=1
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                        "score: {:.4f} ({:.4f})".format(epoch+1, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / count_step, 
                        batch_acc, global_acc / count_step)
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0 and rank == 0) or \
                        global_step == t_total:
                    save_checkpoint(model, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training and rank == 0: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        test_result = test(args, model, val_dataset, rank = rank)
                        eval_result = evaluate(val_dataset, test_result)
                        rank_accs = eval_result['i2t_retrieval']
                        if rank_accs['R@1'] > best_score:
                            best_score = rank_accs['R@1']
                        epoch_log = {'epoch': epoch+1, 'global_step': global_step, 
                                     'R1': rank_accs['R@1'], 'R5': rank_accs['R@5'], 
                                     'R10': rank_accs['R@10'], 'best_R1':best_score}
                        log_json.append(epoch_log)
                        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                            json.dump(log_json, fp) 
    return global_step, global_loss / global_step


def test(args, model, eval_dataset, rank=None):
    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # args.eval_batch_size = args.per_gpu_eval_batch_size
    args.eval_batch_size = 20
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    eval_sampler = SequentialSampler(eval_dataset)
    
    image_loader = DataLoader(eval_dataset, sampler=eval_sampler, pin_memory=True,
            batch_size=1, num_workers=args.num_workers)
    
    text_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, 
                             sampler=SequentialSampler(eval_dataset),
                             num_workers=args.num_workers, pin_memory=True)
    
    text_preload = list()
    for _, _b in tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append({
            'input_ids': _b[0].to(rank),
            'attention_mask': _b[1].to(rank),
            'token_type_ids': _b[2].to(rank),
        })
    
    image_preload = list()
    for _, _b in tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append(_b[3].to(rank))
    
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    
    for idx, img_batch in tqdm(enumerate(image_preload), desc='rank loop'):
        _result = []
        _img_feat = img_batch
        _, l, c = _img_feat.shape

        labels = [0] * len(image_preload)
        labels[idx] = 1
        labels = np.reshape(labels, [len(text_preload), -1])
                
        for i, txt_batch in enumerate(text_preload):
            fblen = len(txt_batch['input_ids'])
            img_feat = _img_feat.expand(fblen, l, c)
            label = labels[i]
            label = torch.Tensor(label).long().to(rank)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    inputs = {
                            'input_ids':      txt_batch['input_ids'],
                            'attention_mask': txt_batch['attention_mask'],
                            'token_type_ids': txt_batch['token_type_ids'],
                            'img_feats':      img_feat,
                            'labels':         label
                            }
                    _, logits = model(**inputs)[:2]
                    if args.num_labels == 2:
                        probs = softmax(logits)
                        result = probs[:, 1] # the confidence to be a matched pair
                    else:
                        result = logits
                    result = [_.to(torch.device("cpu")) for _ in result]
                    _result.extend(result)        
        results.update({idx: _result})
    return results

def compute_ranks(dataset, results):
    # labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    # labels = [1] + [0]* (len(dataset) - 1)
    similarities = np.array([results[i] for i in range(len(dataset))])
    labels = np.zeros_like(similarities)
    np.fill_diagonal(labels, 1)
    num_dim = len(dataset)
    # labels = np.reshape(np.array(labels * num_dim), [-1, num_dim])
    # similarities = np.reshape(similarities, [-1, num_dim])
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_dim
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        t2i_ranks.append(rank)
        
    labels = np.swapaxes(labels, 0, 1)
    similarities = np.swapaxes(similarities, 0, 1)
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_dim
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)
    return i2t_ranks, t2i_ranks

def evaluate(eval_dataset, test_results):
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ <= r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ <= r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result


def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length', 
            'max_img_seq_length', 'add_od_labels', 'od_label_type',
            'use_img_layernorm', 'img_layer_norm_eps']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def main(rank, args, world_size, seed):
    mp.set_start_method('fork', force=True)
    ddp_setup(rank, world_size)
    set_seed(seed, world_size)
    logger.warning("Device: %s, n_gpu: %s", rank+1, world_size)
    if rank ==0:
        logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))
 
    config_class, tokenizer_class = BertConfig, BertTokenizer
    model_class = ImageBertForSequenceClassification
    if args.do_train:
        config = config_class.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=args.num_labels, finetuning_task='ir')
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.img_layer_norm_eps = args.img_layer_norm_eps
        config.use_img_layernorm = args.use_img_layernorm
        model = model_class.from_pretrained(args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    if rank == 0:
        logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataset = RetrievalDataset(args, tokenizer, 'train', is_train=True)
        if args.evaluate_during_training:
            val_dataset = RetrievalDataset(args, tokenizer, 'val', is_train=False)
        else:
            val_dataset = None
        global_step, avg_loss = train(args, train_dataset, val_dataset, model, tokenizer, rank=rank)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if args.do_test or args.do_eval:
        args = restore_training_settings(args)
        test_dataset = RetrievalDataset(args, tokenizer, split='test', is_train=False)
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        if rank ==0:
            logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)
        if rank != None:
            model = model.to(rank)
            # wrap with DDP, this step will synch model across all the processes
            model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        elif args.n_gpu > 1 and rank == None:
            model = torch.nn.DataParallel(model)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
        test_result = test(args, model, test_dataset, rank=rank)
        if args.do_eval:
            _ = evaluate(test_dataset, test_result)
            logger.info("*************Finish Evaluate*****************")       
    destroy_process_group()

if __name__ == "__main__":
    args = get_args() 
    world_size = torch.cuda.device_count()
    args.n_gpu = world_size
    seed = 1234
    mp.spawn(main, args=(args, world_size, seed), nprocs=world_size, join=True)
