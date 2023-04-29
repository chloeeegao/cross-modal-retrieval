import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision 
from torchvision import transforms, models
from torchvision.models import vit_b_16, vit_b_32, resnet50, wide_resnet50_2
import timm
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter

from transformers import ViTConfig, ViTModel
from torch.utils.data import DataLoader


import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAP_LOC = None if torch.cuda.is_available() else 'cpu'


def get_image_model(name, n_classes):
    
    if "vit" in name:
        model = timm.create_model(name, pretrained=True)
        num_features = model.head.in_features
    else:
        model = globals()[name](pretrained=True)
        num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_classes)

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
def eval(test_data, model, loss_fn, total_test_step, batch_size):
    
    model.eval()
    test_loss = 0
    test_accuracy = 0

    test_data_size = len(test_data)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size = batch_size)
    
    with torch.no_grad():
        for batch_data in test_dataloader:
            imgs, targets = batch_data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            test_accuracy += accuracy
            
    total_test_step += 1
    
    return total_test_step, test_accuracy/test_data_size, test_loss/len(test_dataloader)


def train(model, loss_fn, optimizer, n_epochs, train_data, test_data, batch_size):
    
    writer = SummaryWriter('./output/food101_logs')
    global_step = 0
    total_test_step = 0
    best_accuracy = 0
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size = batch_size)
    
    for i in range(n_epochs):
        
        model.train()
        for batch_data in train_dataloader:
            imgs, targets = batch_data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            
            loss=loss_fn(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            if global_step % 20 == 0:               
                t = time.localtime()
                print("{}, Epoch: {}, Global Step: {}, Loss:{}".format(time.asctime(t), 
                                                                            i+1, global_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), global_step)
            
        total_test_step, test_accuracy, test_loss = eval(test_data, model, loss_fn, total_test_step, batch_size)
        
        print("Test Loss: {}".format(test_loss))
        print("Test Accuracy: {}".format(test_accuracy))
        writer.add_scalar("test_loss", test_loss, total_test_step)
        writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
        
        total_test_step = total_test_step
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model, "./output/models/food101_{}.pth".format(i+1))
            print("model saved")
    
    writer.close()
        
        
def main():
    
    transforms_train = transforms.Compose([transforms.Resize((256)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))])


    transforms_val = transforms.Compose([transforms.Resize((256)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))])
        
    train_data = torchvision.datasets.Food101(root="/data/s2478846/git_repo/cross-modal-retrieval/data", split='train', transform=transforms_train,
                                          download=True)
    test_data=torchvision.datasets.Food101(root="/data/s2478846/git_repo/cross-modal-retrieval/data", split='test', transform=transforms_val,
                                          download=True)
    
    model = get_image_model(name='wide_resnet50_2', n_classes=101)
    print("parameters: {}".format(count_parameters(model)))
    
    if device != 'cpu' and torch.cuda.device_count() > 1:
        ngpus = torch.cuda.device_count()
        model = nn.DataParallel(model)
    model = model.to(device)
    if device != 'cpu':
        cudnn.benchmark = True
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    n_epochs = 100
    batch_size = 256
    train(model, loss_fn, optimizer, n_epochs, train_data, test_data, batch_size)
    
if __name__=='__main__':
    main()