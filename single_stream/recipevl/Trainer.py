import os
import numpy as np
import random
import pickle
import torch
import json
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling, 
    DataCollatorForWholeWordMask, 
)
from dataset import collate
from module.objectives import compute_irtr_recall
from module.utils import make_dir, count_parameters, setup_logger
from config import get_args

# create logger 
args = get_args()
global logger
save_path = os.path.join(args.output_dir, args.model_name)
make_dir(save_path)
logger = setup_logger("recipevl", save_path, 'log.txt')


# class metric_log:
    
#     def __init__(self) -> None:
#         pass
    
#     def 


class Trainer:
    def __init__(self, args, model, train_dataset, test_dataset=None, rank=None):
        self.config = vars(args)
        self.rank = rank
        if self.rank != None:
            self.local_rank = rank
            logger.info(f"Running on rank {rank+1} / {dist.get_world_size()}.")
        self.tokenizer = train_dataset.tokenizer
        fast_run = self.config['fast_run']
        # build dataloader
        if not fast_run:
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
        else:
            if self.local_rank == 0:
                logger.info('****** FAST RUN *************')
            self.train_dataset = [train_dataset[i] for i in range(100)]
            self.test_dataset = [test_dataset[i] for i in range(100)]
            self.config['n_epochs'] = 2
        
        self.train_loader = self.prepare_dataloader(self.train_dataset)
        self.test_loader = self.prepare_dataloader(self.test_dataset) if test_dataset else None 
        
        # initialize train 
        self.epochs_run = 0
        self.model = model
        self.total_params = count_parameters(self.model)
        
        if self.config['resume_from'] != '':
            self.optimizer = self.load_checkpoint()
        else:  
            self.optimizer, self.scheduler = self.set_scheduler()
        
        if self.rank != None:
            self.model = model.to(self.local_rank)
            # wrap with DDP, this step will synch model across all the processes
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)  #module
        # elif self.rank == None and self.config['do_train']:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     if device != 'cpu' and torch.cuda.device_count() > 1:
        #         self.model = nn.DataParallel(model)
        #     self.model = self.model.to(device)
        #     self.local_rank = device
        # else:
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     self.model = self.model.to(device)
        #     self.local_rank = device
        
        # print training information 
        if self.config['do_train'] and self.local_rank == 0:
            logger.info("*************** Running training ****************")
            t_total =len(self.train_loader) // self.config['gradient_accumulation_steps'] \
                    * self.config['n_epochs']
            train_batch_size = self.config['per_gpu_train_batch_size'] * max(1, self.config['n_gpus'])
            logger.info("  Num examples = %d", len(self.train_dataset))
            logger.info("  Num Epochs = %d", self.config['n_epochs'])
            logger.info("  Batch size per GPU = %d", self.config['per_gpu_train_batch_size'])
            logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                        train_batch_size * self.config['gradient_accumulation_steps'])
            logger.info("  Gradient Accumulation steps = %d", self.config['gradient_accumulation_steps'])
            logger.info("  Total optimization steps = %d", t_total)
            logger.info("  Total parameters = %d", self.total_params)
      
    def prepare_dataloader(self, dataset):
    
        collator = (
                DataCollatorForWholeWordMask
                if self.config['whole_word_masking']
                else DataCollatorForLanguageModeling
            )
        
        mlm_collator = collator(tokenizer=self.tokenizer, mlm=True, mlm_probability= self.config['mlm_prob'])
        
        if self.config['do_train']:
            batch_size = self.config['per_gpu_train_batch_size'] * self.config['n_gpus']
        else:
            batch_size = self.config['eval_batch_size']
        
        if self.rank != None:
            loader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                num_workers=self.config['num_workers'],
                                sampler=DistributedSampler(dataset),
                                collate_fn=lambda batch: collate(batch, mlm_collator),
                                pin_memory=True,
                                drop_last=False)        
        else:
            loader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                num_workers=self.config['num_workers'],
                                collate_fn=lambda batch: collate(batch, mlm_collator),
                                pin_memory=True,
                                drop_last=False)

        return loader
        
    def run_batch(self, batch, is_train=True):
        
        output = self.model(batch)
        batch_loss = sum([v for k, v in output.items() if "loss" in k])
        batch_acc = output['itm_accuracy'].item()
        
        if is_train:
            batch_loss.backward()
            
        return batch_loss.item(), batch_acc
        
        
    def run_epoch(self, epoch, dataloader, is_train=True):
        
        step_type = "Train" if is_train else "Val"
        
        for step, batch in enumerate(dataloader):
            batch_loss, batch_acc = self.run_batch(batch, is_train)
            if is_train and (step+1) % self.config['gradient_accumulation_steps'] == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            if step % self.config['logging_steps'] == 0:
                logger.info(f"[GPU{self.local_rank}] Epoch {epoch} | Step {step}| lr {self.optimizer.param_groups[0]['lr']:.8f}| {step_type} Loss {batch_loss:.5f} | score {batch_acc: .4f}")     
        return step
    
    def train(self):
        global_step = 0
        best_score = 0.0
        log_json = []
        
        for epoch in range(self.epochs_run, self.config['n_epochs']):
            epoch +=1
            step = self.run_epoch(epoch, self.train_loader, is_train=True)
         
            if self.config['evaluate_during_training']:
                global_step += step
                if self.local_rank == 0:
                    logger.info("Perform evaluation at step: %d", global_step)
                    logger.info("Num examples = %d", len(self.test_dataset))
                    logger.info("Evaluation batch size = %d", self.config['eval_batch_size'])
                self.model.eval()
                _ = self.run_epoch(epoch, self.test_loader, is_train=False)
                (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(self.model, self.test_dataset, self.tokenizer)
                if ir_r1.item() > best_score:
                    logger.info("Updating best checkpoints")
                    best_score = ir_r1.item()
                    if self.local_rank == 0:
                        self.save_model(epoch, global_step)
                epoch_log = {'epoch': epoch, 'global_step': global_step, 
                'R1': ir_r1.item(), 'R5': ir_r5.item(), 
                'R10': ir_r10.item(), 'best_R1':best_score}
                log_json.append(epoch_log)
                with open(self.config['output_dir']+self.config['model_name']+'/eval_logs.json', 'w') as fp:
                    json.dump(log_json, fp) 
                logger.info(f"I2T Retrieval: {ir_r1.item():.4f} @R1, {ir_r5.item():.4f} @R5, {ir_r10.item():.4f} @R10| T2I Retrieval: {tr_r1.item():.4f} @R1, {tr_r5.item():.4f} @R5, {tr_r10.item():.4f} @R10")
        if self.local_rank == 0:
            logger.info(f"**************** Finish training at Epoch {epoch} *******************")

    def eval(self, test_dataset, N=100, K=1):
        logger.info("******* Running evaluation *********")
        logger.info("Num examples = %d", len(test_dataset))
        ir_r1_l, ir_r5_l, ir_r10_l, tr_r1_l, tr_r5_l, tr_r10_l = [],[],[],[],[],[]
        index = [i for i in range(len(test_dataset))]
        for _ in range(K):
            random_sample = random.sample(index, N)
            dataset = [test_dataset[n] for n in random_sample]
            (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(self.model, dataset, self.tokenizer)
            ir_r1_l.append(ir_r1)
            ir_r5_l.append(ir_r5)
            ir_r10_l.append(ir_r10)
            tr_r1_l.append(tr_r1)
            tr_r5_l.append(tr_r5)
            tr_r10_l.append(tr_r10)
        ir_r1_mean = np.mean(ir_r1_l)
        ir_r5_mean = np.mean(ir_r5_l)
        ir_r10_mean = np.mean(ir_r10_l)
        tr_r1_mean = np.mean(tr_r1_l)
        tr_r5_mean = np.mean(tr_r5_l)
        tr_r10_mean = np.mean(tr_r10_l)
        logger.info(f"TEST I2T Retrieval: {ir_r1_mean:.4f} @R1, {ir_r5_mean:.4f} @R5, {ir_r10_mean:.4f} @R10| TEST T2I Retrieval: {tr_r1_mean:.4f} @R1, {tr_r5_mean:.4f} @R5, {tr_r10_mean:.4f} @R10")
        logger.info("************* Finish evaluation *****************")

    def save_model(self, epoch, global_step):
        
        checkpoint_dir = os.path.join(self.config['output_dir'],self.config['model_name'], 'checkpoint-{}-{}'.format(epoch, global_step))
        make_dir(checkpoint_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_state_dict = model_to_save.state_dict()
        torch.save({'model_state_dict': model_state_dict, 'optim_state_dict': self.optimizer.state_dict()},
                    os.path.join(checkpoint_dir,'model_optim.pt'))
    
        #save config
        self.config['curr_epoch'] = epoch
        pickle.dump(self.config, open(os.path.join(checkpoint_dir, 'config.pkl'), 'wb'))
        logger.info("Save checkpoint to {}".format(checkpoint_dir))
        
    def load_checkpoint(self):

        logger.info(f"Resuming training from: {self.config['resume_from']}")
        resume_path = os.path.join(self.config['output_dir'], self.config['model_name'], self.config['resume_from'])
        checkpoint = torch.load(os.path.join(resume_path, 'model_optim.pt'))
        model_state_dict = checkpoint['model_state_dict']
        opt_state_dict = checkpoint['optim_state_dict']
        config = pickle.load(open(os.path.join(resume_path, 'config.pkl'), 'rb'))
        
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
            
        optimizer, _ = self.set_scheduler()
        optimizer.load_state_dict(opt_state_dict)
        self.epochs_run = config['curr_epoch']
        # logger.info(f"Resuming training from Epoch {self.epochs_run}")     
        return optimizer
    
    def set_scheduler(self):
        
        lr = self.config['learning_rate']
        wd = self.config['weight_decay']
        # Prepare optimizer and scheduler
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not \
                any(nd in n for nd in no_decay)], 'weight_decay': wd},
            {'params': [p for n, p in self.model.named_parameters() if \
                any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(grouped_parameters, lr=lr, eps=self.config['adam_epsilon'])
        
        if self.config['max_steps'] is None:
            max_steps = (
                len(self.train_loader) // self.config['gradient_accumulation_steps'] \
                * self.config['n_epochs']
            )
        else:
            max_steps = self.config['max_steps']
        
        if self.config['scheduler_name'] == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                        optimizer, num_warmup_steps=self.config['warmup_steps'], num_training_steps=max_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(
                        optimizer, num_warmup_steps=self.config['warmup_steps'], num_training_steps=max_steps)
        
        return optimizer, scheduler
  
        
       