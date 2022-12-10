from asyncio.log import logger
from oscar.utils.misc import mkdir
import torch
import torch.nn as nn
import torchvision 
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import os
import pickle
import random

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, wide_resnet50_2
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAP_LOC = None if torch.cuda.is_available() else 'cpu'


class foodimages(Dataset):
    def __init__(self, root, split='train', transform=None, mode ='train'):
         
        self.mode = mode 
        self.transform = transform
        self.root = root
        self.split = split
        
        with open(os.path.join(root, 'classes1M.pkl'),'rb') as f:
            self.class_dict = pickle.load(f)
            self.id2class = pickle.load(f)

        self.num_class = len(set(self.id2class.keys()))+1
        
        self.train_imgs = dict()
        self.test_imgs = dict()
        
        if self.mode == 'train':
            i = 0
            data = pickle.load(open(os.path.join(root,'traindata', self.split + '.pkl'), 'rb'))
            recipe_ids = list(data.keys())
            for rep_id in recipe_ids:
                img_names = data[rep_id]['images']
                if self.class_dict[rep_id] !=0:
                    label = self.class_dict[rep_id]
                    if len(img_names) <= 5:
                        for img in img_names:
                            self.train_imgs[i] = (img, label, self.split)
                            i += 1
                    else:
                        samples = random.sample(img_names, 5)
                        for img in samples:
                            self.train_imgs[i] = (img, label, self.split)
                            i += 1
        else:
            j = 0
            splits = ['train', 'val', 'test']
            for split in splits:
                data = pickle.load(open(os.path.join(root,'traindata', split + '.pkl'), 'rb'))
                recipe_ids = list(data.keys())
                for rep_id in recipe_ids:
                    img_names = data[rep_id]['images']
                    if self.class_dict[rep_id] ==0:
                        for img in img_names:
                            self.test_imgs[j] = (img, split)
                            j += 1
                    
    def __len__(self):
        if self.mode =='train':
            return len(self.train_imgs)
        else:
            return len(self.test_imgs)
        
    def get_image(self, img_name, split):
        img_name = '/'.join(img_name[:4])+'/'+img_name
        img = Image.open(os.path.join(self.root, split, img_name))
        if self.transform is not None:
            img = self.transform(img)
        return img
        
    def __getitem__(self, index):
        
        if self.mode == 'train':
            img, label, split = self.train_imgs[index] 
            img = self.get_image(img, split)
            return img, label
        else:
            img, split = self.test_imgs[index]
            img = self.get_image(img, split)
            return img
    

def get_loader(root, batch_size, resize, im_size, drop_last=False, mode='train'):
    
    ## resize = 256 im_size = 224
    
    transforms_list1 = [transforms.Resize((resize))]
    transforms_list2 = [transforms.Resize((resize))]

    # Image preprocessing, normalization for pretrained resnet
    transforms_list1.append(transforms.RandomHorizontalFlip())
    transforms_list1.append(transforms.RandomCrop(im_size))
    transforms_list1.append(transforms.ToTensor())
    transforms_list1.append(transforms.Normalize((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225)))
    
    # for evaluation 
    transforms_list2.append(transforms.CenterCrop(im_size))
    transforms_list2.append(transforms.ToTensor())
    transforms_list2.append(transforms.Normalize((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225)))


    transforms_ = transforms.Compose(transforms_list1)
    transforms__ = transforms.Compose(transforms_list2)
    
    train_dataset = foodimages(root, split='train', transform=transforms_, mode=mode)
    val_dataset = foodimages(root, split='val', transform=transforms__, mode=mode)
    test_dataset = foodimages(root, split='test', transform=transforms__, mode=mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=16,drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=16,drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=16,drop_last=drop_last)
    
    return train_loader, val_loader, test_loader
    

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, image_model='wide_resnet50_2', pretrained=False):
        super(ImageClassifier, self).__init__()
        self.image_model = image_model
        
        # load wide_resnet50_2 pretrained on food101
        backbone = globals()[image_model](pretrained=pretrained)
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        in_feats = backbone.fc.in_features

        self.fc = nn.Linear(in_feats, num_classes)
        # self.linear = nn.Linear(1024, num_classes)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, images, freeze_backbone=True):
        """Extract feature vectors from input images."""
        if not freeze_backbone:
            feats = self.backbone(images)
        else:
            with torch.no_grad():
                feats = self.backbone(images)
        feats = feats.view(feats.size(0), feats.size(1),
                           feats.size(2)*feats.size(3))

        feats = torch.mean(feats, dim=-1)
        out = self.fc(feats)
        # out = self.linear(self.dropout(nn.ReLU()(feature)))
        
        return nn.Tanh()(out)
        
    
def eval(model, test_loader):
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for batch in test_loader:  
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            accuracy += (predicted == labels).sum().item()
    
    accuracy = (100 * accuracy/total)
    return accuracy
    
    
def train(model, train_loader, val_loader, num_epochs):
    
    global logger
    output_dir = './output/'
    mkdir(output_dir)
    checkpoint_dir = os.path.join(output_dir, 'best_checkpoint')
    mkdir(checkpoint_dir)
    logger = setup_logger("image_classifer", output_dir, 0)
    
    t_total = len(train_loader) * num_epochs
    
    logger.info("***** Running training *****")
    logger.info("  Num batches = %d", len(train_loader))
    logger.info("  Num Epochs = %d", num_epochs)
    logger.info("  Total optimization steps = %d", t_total)
    
    best_accuracy = 0.0
    global_step = 0
    global_loss = 0.0
    # model.zero_grad()
    model.to(device)
    
    params_fc = list(model.fc.parameters()) 
        # + list(model.linear.parameters())
    
    loss_fn =nn.CrossEntropyLoss()
    optimizer = Adam(params_fc, lr=0.0001, weight_decay=0.01)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0.0
        # accuracy = 0.0
        
        for step, (imgs, labels) in enumerate(train_loader):
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # compute output
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            # loss = loss.to(device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
                        
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            running_acc += (predicted == labels).sum().item()
            running_acc = (100 * running_acc/total)
            global_step += 1
            global_loss += loss.item()
            # if (step + 1) % 1 == 0:
            #     # compute gradient and do SGD step
                
            #     # scheduler.step()
            #     model.zero_grad()
            
            # print every 50 steps
            if step % 50 == 99:
                 logger.info('Epoch: {}, global_step:{}, lr:{:.6f}, loss:{:.4f}, accuracy:{:.6f}'.format(
                     epoch+1, global_step, optimizer.param_groups[0]['lr'], running_loss/100, running_acc))
                 
                 running_loss = 0.0 
                 running_acc = 0.0
                 total = 0
                 
                 
        accuracy = eval(model, val_loader)
        logger.info('Epoch: {}, test accuracy: {:.6f}'.format(epoch+1, accuracy))
    
        if accuracy > best_accuracy:
            # checkpoint_dir = os.path.join(output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_best.pkl'))
            logger.info("Updating best checkpoint to {}".format(checkpoint_dir))
            best_accuracy = accuracy
            logger.info("Best accuracy: {}".format(best_accuracy))
    
    return global_step, global_loss/ global_step

def main():
    root = '/data/s2478846/data'
    batch_size = 256
    resize = 256
    im_size = 224
    num_epochs = 50
    dataset = foodimages(root)
    num_class = dataset.num_class
    model = ImageNet(num_classes=num_class,image_model='resnet50')
    train_loader, val_loader, test_loader = get_loader(root, batch_size, resize, im_size, 
                                                        drop_last=False, mode='train')
    gloabl_step, accuracy = train(model, train_loader, val_loader, num_epochs=num_epochs)
    logger.info("Training done: total_step = %s, accuracy=%s", gloabl_step, accuracy)
    
if __name__ == '__main__':
    main()