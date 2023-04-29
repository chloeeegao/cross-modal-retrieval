import os
import random
from collections import defaultdict
import numpy as np
import pickle
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_loader
from config import get_args
from model import get_model
from eval import computeAverageMetrics
from utils.utils import *
from utils.loss import TripletLoss
from utils.logger import setup_logger


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAP_LOC = None if torch.cuda.is_available() else 'cpu'

def trainIter(args, split, loader, model, optimizer,
              loss_function, scaler, metrics_epoch):
    
    with autocast():
        _, title, act_ing, img = next(loader)

        img = img.to(device) if img is not None else None
        title = title.to(device)
        act_ing= act_ing.to(device)

        if split=='val':
            with torch.no_grad():
                img_feat, recipe_feat = model(title, act_ing, img)
        else:
            img_feat, recipe_feat = model(title, act_ing, img, freeze_backbone= args.freeze_backbone)

        # loss_paired = 0
        # # if img is not None:
        #     loss_paired = loss_function(img_feat, recipe_feat)
        #     metrics_epoch['loss_paired'].append(loss_paired.item())

        loss = loss_function(img_feat, recipe_feat)
        metrics_epoch['loss'].append(loss.item())

    if split == 'train':
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if img_feat is not None:
        img_feat = img_feat.cpu().detach().numpy()

    recipe_feat = recipe_feat.cpu().detach().numpy()

    return img_feat, recipe_feat, metrics_epoch


def train(args):

    checkpoints_dir = os.path.join(args.save_dir, args.model_name)
    make_dir(checkpoints_dir)

    if args.tensorboard:
        writter = SummaryWriter(checkpoints_dir)    

    loaders = {}

    if args.resume_from != '':
        print('resuming from checkpoint:', args.resume_from)
        vars_to_replace = ['batch_size', 'tensorboard', 'model_name',
                            'lr', 'scale_lr', 'freeze_backbone',
                            'load_optimizer']

        store_dict = {}

        for var in vars_to_replace:
            store_dict[var] = getattr(args, var)

        resume_path = os.path.join(args.save_dir, args.resume_from)
        args, model_dict, optim_dict = load_checkpoint(resume_path,'curr',MAP_LOC, store_dict)

        # load current state of training

        curr_epoch = args.curr_epoch
        best_loss = args.best_loss

        for var in vars_to_replace:
            setattr(args, var, store_dict[var])

    else:
        curr_epoch = 0
        best_loss = 0
        model_dict, optim_dict = None, None

    for split in ['train', 'val']:
        loader, dataset = get_loader(args.root, args.batch_size,
                            args.resize, args.imsize, augment=True,
                            split=split, mode=split)
        # print('dataset:', len(dataset))
        logger.info(split+' dataset = %d', len(dataset))
        loaders[split] = loader


    vocab_size = len(dataset.get_vocab())
    model = get_model(args, vocab_size)


    params_backbone = list(model.image_encoder.backbone.parameters())
    params_fc = list(model.image_encoder.fc.parameters()) \
                    + list(model.recipe_encoder.parameters())\
                    + list(model.merge_recipe.parameters())

    logger.info("recipe encoder = %d", count_parameters(model.recipe_encoder))
    logger.info("image encoder = %d", count_parameters(model.image_encoder))

    optimizer = get_optimizer(params_fc,
                              params_backbone,
                              args.lr, args.scale_lr, args.wd,
                              freeze_backbone=args.freeze_backbone)

    if model_dict is not None:
        model.load_state_dict(model_dict)
        if args.load_optimizer:
            try:
                optimizer.load_state_dict(optim_dict)
            except:
                print("Could not load optimizer state. Using default initialization...")

    ngpus = 1
    if device != 'cpu' and torch.cuda.device_count() > 1:
        ngpus = torch.cuda.device_count()
        model = nn.DataParallel(model)

    model = model.to(device)

    if device != 'cpu':
        cudnn.benchmark = True

    # learning rate scheduler
    scheduler = get_scheduler(args, optimizer)

    loss_function = TripletLoss(margin=args.margin)
    # training loop
    wait = 0

    scaler = GradScaler()

    for epoch in range(curr_epoch, args.n_epochs):

        for split in ['train', 'val']:
            if split == 'train':
                model.train()
            else:
                model.eval()

            metrics_epoch = defaultdict(list)

            total_step = len(loaders[split])
            loader = iter(loaders[split])
            logger.info('total step = %d', total_step)
            

            img_feats, recipe_feats = None, None


            for i in range(total_step):

                this_iter_loader = loader

                optimizer.zero_grad()
                model.zero_grad()
                img_feat, recipe_feat, metrics_epoch = trainIter(args, split,
                                                                 this_iter_loader,
                                                                 model, optimizer,
                                                                 loss_function,
                                                                 scaler,
                                                                 metrics_epoch)

                if img_feat is not None:
                    if img_feats is not None:
                        img_feats = np.vstack((img_feats, img_feat))
                        recipe_feats = np.vstack((recipe_feats, recipe_feat))
                    else:
                        img_feats = img_feat
                        recipe_feats = recipe_feat

                if i != 0 and i % args.log_every == 0: # not args.tensorboard and
                    # log metrics to stdout every few iterations
                    avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}
                    text_ = "split: {:s}, epoch [{:d}/{:d}], step [{:d}/{:d}]"
                    values = [split, epoch, args.n_epochs, i, total_step]
                    for k, v in avg_metrics.items():
                        text_ += ", " + k + ": {:.4f}"
                        values.append(v)
                    str_ = text_.format(*values)
                    logger.info(str_)

            # computes retrieval metrics (average of 10 runs on 1k sized rankings)
            retrieval_metrics = computeAverageMetrics(img_feats, recipe_feats,
                                                      1000, 10, forceorder=True)

            for k, v in retrieval_metrics.items():
                metrics_epoch[k] = v

            avg_metrics = {k: np.mean(v) for k, v in metrics_epoch.items() if v}
            # log to stdout at the end of the epoch (for both train and val splits)
            # if not args.tensorboard:
            text_ = "AVG. split: {:s}, epoch [{:d}/{:d}]"
            values = [split, epoch, args.n_epochs]
            for k, v in avg_metrics.items():
                text_ += ", " + k + ": {:.4f}"
                values.append(v)
            str_ = text_.format(*values)
            logger.info(str_)

            # log to tensorboard at the end of one epoch
            if args.tensorboard:
                # 1. Log scalar values (scalar summary)
                for k, v in metrics_epoch.items():
                    writter.add_scalar('{}/{}'.format(split, k), np.mean(v), epoch)

        # monitor best loss value for early stopping
        # if the early stopping metric is recall (the higher the better),
        # multiply this value by -1 to save the model if the recall increases.
        if args.es_metric.startswith('recall'):
            mult = -1
        else:
            mult = 1

        curr_loss = np.mean(metrics_epoch[args.es_metric])

        if args.lr_decay_factor != -1:
            if args.scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(curr_loss)
            else:
                scheduler.step()

        if curr_loss*mult < best_loss:
            # if not args.tensorboard:
            logger.info("Updating best checkpoint")
            save_model(model, optimizer, 'best', checkpoints_dir, ngpus)
            best_loss = curr_loss*mult

            wait = 0
        else:
            wait += 1

        # save current model state to be able to resume it
        save_model(model, optimizer, 'curr', checkpoints_dir, ngpus)
        args.best_loss = best_loss
        args.curr_epoch = epoch
        pickle.dump(args, open(os.path.join(checkpoints_dir,
                                            'args.pkl'), 'wb'))

        if wait == args.patience:
            break

    if args.tensorboard:
        writter.close()


def main():
    args = get_args()
    global logger
    make_dir(args.output_dir)
    logger = setup_logger(args.model_name, args.output_dir, 0, filename=args.model_name+args.date+'log.txt')
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    train(args)


if __name__ == "__main__":
    main()