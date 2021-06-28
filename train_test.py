from numpy import argmax
from sklearn.metrics import accuracy_score
from numpy import vstack
from dataset import MyDataset
from slowfastnet import SlowFast,Bottleneck
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import time
import wandb
import json



def train_model(model, criterion, optimizer, scheduler,trainLoader ,num_epochs=25):
    since = time.time()
    wandb.init(project="kids-model")
    device = torch.device("cpu")
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training phase
        for phase in ['train']:
            if phase=='train':
                model.train()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            for inputs, labels in trainLoader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

        epoch_loss = running_loss / 841 #should this be 840(40videos * 21 classes) or just40
        epoch_acc = running_corrects.double() / 841

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        wandb.log({'accuracy': epoch_loss, 'loss': epoch_loss})
            # deep copy the model


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    #
    # # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model

def getClassDict(jsonfile):
    with open(jsonfile) as json_file:
        data = json.load(json_file)

        return data

def mapIntToClass(prediction,classjson):
    class_dict = getClassDict(classjson)
    keys_labels = list(class_dict.keys())
    values_int = list(class_dict.values())
    position = values_int.index(prediction)
    label = keys_labels[position]

    return label

def showPredictions(predicted_frames, predicted_target, actual_target,classjson):
    class_dict  = getClassDict(classjson)
    # predicted_frames = predicted_frames.reshape(224,224,3)
    plt.imshow(predicted_frames)
    plt.title(f'Prediction: {predicted_target} - Actual target: {actual_target}')
    plt.show()


# evaluate the model
def evaluate_model(test_dl, model, class_json):
    predictions, actuals = list(), list()
    model.eval()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)


        #top 5: print top-5 predictions
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(yhat)
        pred_classes = preds.topk(k=5).indices[0]

        pred_class_names = []#[mapPrection2Class(str(i)) for i in pred_classes]
        for indx, i in enumerate(pred_classes):
            i = i.detach().numpy()
            pred_class_names.append(mapIntToClass(int(i),class_json))
            print('\t[%s], with probability %.3f.'% (mapIntToClass(int(i),class_json),preds[0][pred_classes[indx]]) )
        # print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))

        #calculate Top-5 accuracy:TODO

        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)

        predicted_label = mapIntToClass(yhat,"className_kidsVal.json")
        actual_y = targets.detach().numpy()
        actual_y = mapIntToClass(int(actual_y[0]),"className_kidsVal.json")

        showPredictions(inputs, predicted_label, actual_y,class_json)

        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc
