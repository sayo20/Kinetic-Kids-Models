"""
In this model, we will finetune slowfast on kids-specific training data and evaluate on the adult specific dataset
"""
import time


from dataset import MyDataset
from slowfastnet import SlowFast,Bottleneck
from torch.utils.data import DataLoader
from train_test import train_model,evaluate_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler




if __name__ == '__main__':

    model = SlowFast(Bottleneck, [3, 4, 6, 3],num_classes=200)


    state_dict = torch.load('slowfast50_best_fixed.pth',map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False #freeze the earlier layers

    num_ftrs = model.fc.in_features #we initialize the model with previous weights and only finetune the last last year(i think)
    model.fc = nn.Linear(num_ftrs, 21)
    # model_ft = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    dataset_train = MyDataset('Dataset Csvs/TrainSplit-kids.csv','className_kidsTrain.json',mode='train')
    train_dataloader = DataLoader(dataset_train,batch_size= 10,shuffle=True)

    dataset_test_adult = MyDataset('Dataset Csvs/ValSplit-adults.csv','className_AdultsVal.json',mode='val')
    test_dataloader_adult = DataLoader(dataset_test_adult,batch_size= 10,shuffle=True)

    dataset_test_kid = MyDataset('Dataset Csvs/ValSplit-kids.csv','className_kidsVal.json',mode='val')
    test_dataloader_kids = DataLoader(dataset_test_kid,batch_size= 10,shuffle=True)

    #train
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler,train_dataloader,
                num_epochs=25)

    #test:kid
    acc_kid = evaluate_model(test_dataloader_kids, model)
    #
    # #test:adult
    acc_adult = evaluate_model(test_dataloader_adult, model)

    #visualize predictions
