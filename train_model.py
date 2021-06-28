# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# Dear Sam,
# Jupyter notebooks are really not as bad as you think! I hope going through this process changes your mind :)

# %%

from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torch.optim import lr_scheduler

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image
# Useful for examining network
from torchsummary import summary
# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14

#custom class
from dataset_ours import MyDataset
from slowfastnet import SlowFast,Bottleneck
from TrainTestCode import train

class MLP(nn.Module):
    def __init__(self,in_dim,sizes,out_dim,nonlin,residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.sizes = sizes
        self.out_dim = out_dim
        self.nonlin = nonlin
        self.residual = residual
        self.in_layer = nn.Linear(in_dim,self.sizes[0])
        self.out_layer = nn.Linear(self.sizes[-1],out_dim)
        self.layers = nn.ModuleList([nn.Linear(sizes[index],sizes[index+1]) for index in range(len(sizes)-1)])


    def forward(self,x):
        x = self.nonlin(self.in_layer(x))

        for index, layer in enumerate(self.layers):
            if ((index % 2) == 0):
                residual = x
                x = self.nonlin(layer(x))
            else:
                x = self.nonlin(residual+layer(x))
        x = self.out_layer(x)
        return x
# Printing out all outputs
InteractiveShell.ast_node_interactivity = 'all'

# %% [markdown]
# # SET UP TARAINING PARAMS (GPU OR CPU)

# %%
batch_size = 32
save_models = "Kid-specificModel.pt"
checkpoint_path = "Kid-specificModel.pth" #saving model
n_classes = 21
# Whether to train on a gpu

device = 'cuda'
print(f'Device: {device}')

# Number of gpus
# %% [markdown]
# # Create Data Loader for training and test split
age ='adults'
# %%
dataset_train_kids = MyDataset(f'Data_Csv/TrainSplit-{age}.csv',f'className_{age}Train.json',mode='train')
dataset_val_kids = MyDataset(f'Data_Csv/ValSplit-{age}.csv',f'className_{age}sVal.json',mode='train')
dataset_test_kids = MyDataset(f'Data_Csv/TestSplit-{age}.csv',f'className_{age}Test.json',mode='train')
dataLoader_kids = {
    'train':DataLoader(dataset_train_kids,batch_size= batch_size,shuffle=True),
    'test': DataLoader(dataset_test_kids,batch_size= batch_size,shuffle=True),
    'val':DataLoader(dataset_val_kids,batch_size= batch_size,shuffle=True)
}
    
#Loader for adult test split because we evaluate model on it as well

dataset_test_adult = MyDataset('Data_Csv/TestSplit-adults.csv','className_AdultsVal.json',mode='val')
test_dataloader_adult = DataLoader(dataset_test_adult,batch_size= batch_size,shuffle=True)
    
    

# %% [markdown]
# # Load Model and freeze previous layers

# %%
model = SlowFast(Bottleneck, [3, 4, 6, 3],num_classes=200)


state_dict = torch.load('slowfast50_best_fixed.pth',map_location=device)#remove map_location on Gpu

model.load_state_dict(state_dict)

for param in model.parameters():
    param.requires_grad = False #freeze the earlier layers

num_ftrs = model.fc.in_features #we initialize the model with previous weights and only finetune the last last year
model.fc =  nn.Linear(num_ftrs, 21)
#model.fc = MLP(num_ftrs,[512,512,512],21,nn.GELU())


model = model.to(device)


# %% [markdown]
# # Configure training loss and optimizer

# %%
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model.fc.parameters(), lr=0.001)
# Decay LR if plateau
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=40, factor=0.5,threshold=0.01)

# %% [markdown]
# # Train Model

# %%
model, history = train(
    model,
    criterion,
    optimizer_ft,
    dataLoader_kids['train'],
    dataLoader_kids['val'],
    exp_lr_scheduler,
    save_models,device = device)

# %% [markdown]
# # INSPECT TRAINING PROGRESS
# create accuracy and loss plots

# %%
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(
        history[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Cross entropy loss')
plt.title('Training and Validation Losses')


# %%
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 * history[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy')
plt.title('Training and Validation Accuracy')

# %% [markdown]
# # Saving the fine-tuned kid-specific model

# %%
torch.save(model.state_dict(), checkpoint_path)

# %% [markdown]
# # Get top-1 and top5 accuracy

# %%
def accuracy(output, target, topk=(1, )):
    """Compute the topk accuracy(s)"""
    if train_on_gpu:
        output = output.to('cuda')
        target = target.to('cuda')

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Find the predicted classes and transpose
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        # For each k, find the percentage of correct
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# %%
#get top-1 and top-5 for kids-test data (do for validation and test combined and then only test)
testiter = iter(dataLoader_kids['test'])

# Get a batch of testing images and labels
features, targets = next(testiter)
print("Top 1 and Top 5 Kids test split")
if train_on_gpu:
    accuracy(model(features.to('cuda')), targets, topk=(1, 5))
else:
    accuracy(model(features), targets, topk=(1, 5))


# %%
#get top-1 and top-5 for adults-test data
testiter = iter(test_dataloader_adult)
# Get a batch of testing images and labels
features, targets = next(testiter)
print("Top 1 and Top 5 Adult test split")
if train_on_gpu:
    accuracy(model(features.to('cuda')), targets, topk=(1, 5))
else:
    accuracy(model(features), targets, topk=(1, 5))

# %% [markdown]
# # Calculate the Accuracy per Class

# %%

def evaluate(model, test_loader, criterion, topk=(1, 5)):
    """Measure the performance of a trained PyTorch model

    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        topk (tuple of ints): accuracy to measure

    Returns
    --------
        results (DataFrame): results for each category

    """

    classes = []
    losses = []
    # Hold accuracy results
    acc_results = np.zeros((len(test_loader.dataset), len(topk)))
    i = 0

    model.eval()
    with torch.no_grad():

        # Testing loop
        for data, targets in test_loader:

            # Tensors to gpu
            if train_on_gpu:
                data, targets = data.to('cuda'), targets.to('cuda')

            # Raw model output
            out = model(data)
            # Iterate through each example
            for pred, true in zip(out, targets):
                # Find topk accuracy
                acc_results[i, :] = accuracy(
                    pred.unsqueeze(0), true.unsqueeze(0), topk)
                
#                 classes.append(model.idx_to_class[true.item()]) #CHANGE THIS LINE CAUSE YOU MAP Differenly but i dont know what true prints
                classes.append(mapIntToClass(int(true.item()),"className_kidsTest.json"))
                # Calculate the loss
                loss = criterion(pred.view(1, n_classes), true.view(1))
                losses.append(loss.item())
                i += 1

    # Send results to a dataframe and calculate average across classes
    results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
    results['class'] = classes
    results['loss'] = losses
    results = results.groupby(classes).mean()

    return results.reset_index().rename(columns={'index': 'class'})

# %%
results = evaluate(model, dataLoader_kids['test'], criterion)
results


# %%
print('Category with minimum accuracy.')
results.loc[results['top1'].idxmin]

# %% [markdown]
# # Make Inference with fine-tuned model
# The method takes in the path to the image we wan to get predictions for

# %%
def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor
def mapIntToClass(prediction,classjson):
    class_dict = getClassDict(classjson)
    keys_labels = list(class_dict.keys())
    values_int = list(class_dict.values())
    position = values_int.index(prediction)
    label = keys_labels[position]

    return label

def predict(image_path, model, topk=5):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns

    """
    real_class = image_path.split('/')[-2]

    # Convert to pytorch tensor
    img_tensor = process_image(image_path)

    # Resize
    if train_on_gpu:
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        
        out = model(img_tensor)
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(out)
        pred_classes = preds.topk(topk).indices[0]

        top_classes = []#[mapPrection2Class(str(i)) for i in pred_classes]
        top_p = []
        for indx, i in enumerate(pred_classes):
            i = i.detach().numpy()
            top_classes.append(mapIntToClass(int(i),"className_kidsVal.json"))
            top_p.append(preds[0][pred_classes[indx]])
            
#         ps = torch.exp(out)
# Model outputs log probabilities
#         # Find the topk predictions
#         topk, topclass = ps.topk(topk, dim=1)

#         # Extract the actual classes and probabilities
#         top_classes = [
#             model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
#         ]
#         top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class


    
    
def display_prediction(image_path, model, topk):
    """Display image and preditions from model"""

    # Get predictions
    img, ps, classes, y_obs = predict(image_path, model, topk)
    # Convert results to dataframe for plotting
    result = pd.DataFrame({'p': ps}, index=classes)

    # Show the image
    plt.figure(figsize=(16, 5))
    ax = plt.subplot(1, 2, 1)
    ax, img = imshow_tensor(img, ax=ax)

    # Set title to be the actual class
    ax.set_title(y_obs, size=20)

    ax = plt.subplot(1, 2, 2)
    # Plot a bar plot of predictions
    result.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)
    plt.xlabel('Predicted Probability')
    plt.tight_layout()
    
    


# %%
#loop over paths in test set and call display_prediction


# %% [markdown]
# # DISPLAY PREDICTIONS BASED ON CATEGORY

# %%

def display_category(model, category, n=4):
    """Display predictions for a category    
    """
    category_results = results.loc[results['class'] == category]
    print(category_results.iloc[:, :6], '/n')

    images = np.random.choice(
        os.listdir(testdir + category + '/'), size=4, replace=False)

    for img in images:
        display_prediction(testdir + category + '/' + img, model, 5)


# %%
#check the results with minimum accuracy
display_category(model, 'anchor')


