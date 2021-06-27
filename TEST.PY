from torch.utils.data.dataloader import DataLoader
from dataset import MyDataset
from slowfastnet import SlowFast,Bottleneck
from torch.utils.data import DataLoader
import torch
from numpy import argmax
from prettytable import PrettyTable
from train_test import train_model,evaluate_model,getClassDict,mapIntToClass,showPredictions


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def count_parameters_(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



def mapPrection2Class(y):
    class_dict = getClassDict("HACS_clips_v1.1__val_dictionary.json")
    vals = class_dict[y]
    path_label = vals[2]
    label = path_label.split("/")[6]

    return label

if __name__ == '__main__':

    model = SlowFast(Bottleneck, [3, 4, 6, 3],num_classes=200)


    state_dict = torch.load('slowfast50_best_fixed.pth',map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)

    dataset_test_kid = MyDataset('Dataset Csvs/ValSplit-kids.csv',"className_kidsVal",mode='val')
    test_dataloader_kids = DataLoader(dataset_test_kid,batch_size= 1,shuffle=True)

    # acc_kid = evaluate_model(test_dataloader_kids, model)


    # print(count_parameters_(model))

    #
    model.eval()
    for batch_ind, batch in enumerate(test_dataloader_kids):
        # batch = batch.cuda()
        img, actual_y = batch
        y =  model(img)
        #top 5: print top-5 predictions
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(y)
        pred_classes = preds.topk(k=5).indices[0]

        pred_class_names = []#[mapPrection2Class(str(i)) for i in pred_classes]
        for indx,i in enumerate(pred_classes):
            i = i.detach().numpy()
            pred_class_names.append(mapPrection2Class(str(i)))
            print('\t[%s], with probability %.3f.'% (mapPrection2Class(str(i)),preds[0][pred_classes[indx]]) )
        # print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))

        # y = y.detach().numpy()
        # y = argmax(y)
        actual_y = actual_y.detach().numpy()
        actual_y = mapIntToClass(int(actual_y[0]),"className_kidsVal.json")
        print("actual y is: ",actual_y)
        # showPredictions(img, pred_class_names, actual_y,"className_kidsVal.json")
        # print("prediction is: ",mapPrection2Class(str(y)))




        pass
