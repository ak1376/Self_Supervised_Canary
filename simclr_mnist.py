# -*- coding: utf-8 -*-
"""
Author: Nikolaos Giakoumoglou
Date: Tue Nov 16 00:33:26 2021

Contrastive Learning Example using the framework SimCLR.
Input images are 28 x 28 x 1 (1 channel) from MNIST dataset. Output is mapped
to 10 classes (numbers 0 to 9).
"""

import sys

sys.path.append('/home/akapoor/MNIST_Classification_Models/')

import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from util import EarlyStopping, MetricMonitor
from util import TwoCropTransform, SupCon, SupConLoss, save_model
from scipy.stats import ks_2samp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np 

class Encoder(torch.nn.Module):
    "Encoder network"
    def __init__(self):
        super(Encoder, self).__init__()
        # L1 (?, 28, 28, 1) -> (?, 28, 28, 32) -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(p=0.2)
            )
        # L2 (?, 14, 14, 32) -> (?, 14, 14, 64) -> (?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(p=0.2)
            )
        # L3 (?, 7, 7, 64) -> (?, 7, 7, 128) -> (?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            # torch.nn.Dropout(p=0.2)
            )
        self._to_linear = 4 * 4 * 128

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # Flatten them for FC
        return x
    
temperature = 1
batch_size = 64
device = 'cuda'

def pretraining(epoch, model, contrastive_loader, optimizer):
    "Contrastive pre-training over an epoch"
    metric_monitor = MetricMonitor()
    model.train()
    negative_similarities_for_epoch = []
    ntxent_positive_similarities_for_epoch = []
    mean_pos_cos_sim_for_epoch = []
    mean_ntxent_positive_similarities_for_epoch = []
    # for batch_idx, (data,labels) in enumerate(contrastive_loader):
    for batch_idx, ((data1, labels1), (data2, labels2)) in enumerate(contrastive_loader):
        # data = torch.cat([data[0], data[1]], dim=0)
        data = torch.cat((data1,data2), dim = 0)
        labels = labels1
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
        bsz = labels.shape[0]
        data = data.to(torch.float32)
        features = model(data)
        batch_size = int(features.shape[0]/2)
        
        # L2-normalize each row
        normalized_features = features / features.norm(dim=1, keepdim=True)
        
        # f1, f2 = torch.split(normalized_features, [bsz, bsz], dim=0)
        # normalized_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                
        similarity_scores = torch.div(
            torch.matmul(normalized_features, normalized_features.T),
            temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(similarity_scores, dim=1, keepdim=True)
        logits = similarity_scores - logits_max.detach()
        
        exp_logits = torch.exp(logits)
        all_but_identity_mask = 1-torch.eye(logits.shape[0]).to(device)
        
        # tile mask
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        
        positive_mask = mask * logits_mask
        negative_mask = logits_mask*(1-mask)
        
        # Let's find the loss for positive samples. 
        # First find the exp_logits for each row
        pos_logits = torch.sum(positive_mask*exp_logits, dim = 1)
        
        # Find the sum of the logits for all samples except for identity
        all_other_logits = torch.sum(exp_logits*all_but_identity_mask, dim = 1)
        
        loss_ij = -1*torch.log(torch.div(pos_logits, all_other_logits))
        
        batch_loss = torch.mean(loss_ij)
        loss = batch_loss
        # Now extract the similarity scores for the positive and negative samples
        
        positive_similarities = positive_mask*similarity_scores
        positive_similarities = positive_similarities[positive_similarities!=0]
        
        a = similarity_scores*negative_mask
        i, j = np.triu_indices(a.shape[0], k=1)
        upper_triangle_values = a[i, j]
        
        # Any value of upper_triangle_values that equals 0 means that it belongs to the masked index
        negative_similarities = upper_triangle_values[upper_triangle_values !=0]
        
        
        # normalized_features = F.normalize(features, p = 2, dim = 0)
        # f1, f2 = torch.split(normalized_features, [bsz, bsz], dim=0)
        # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # if method == 'SupCon':
        #     loss = criterion(features, labels)
        # elif method == 'SimCLR':
        #     loss, negative_similarities, positive_similarities = criterion(features)
        # else:
        #     raise ValueError('contrastive method not supported: {}'.format(method))
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Learning Rate", optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        negative_similarities_for_epoch.append(negative_similarities)
        ntxent_positive_similarities_for_epoch.append(positive_similarities)
        
        # # Calculate the mean cosine similarity of the model feature representation for the positive pairs.
        # # Slice the tensor to separate the two sets of features you want to compare
        f1, f2 = torch.split(normalized_features, [bsz, bsz], dim=0)
        normalized_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        tensor_a = normalized_features[:, 0, :].clone().detach()
        tensor_b = normalized_features[:, 1, :].clone().detach()
        
        # Compute the cosine similarities
        similarities = F.cosine_similarity(tensor_a, tensor_b, dim=1)
        mean_pos_cos_sim_for_batch = torch.mean(similarities).clone().detach().cpu().numpy()
        mean_ntxent_positive_similarities = torch.mean(positive_similarities).clone().detach().cpu().numpy()
        mean_ntxent_positive_similarities_for_epoch.append(float(mean_ntxent_positive_similarities))
        mean_pos_cos_sim_for_epoch.append(float(mean_pos_cos_sim_for_batch))

    print("[Epoch: {epoch:03d}] Contrastive Pre-train | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], negative_similarities_for_epoch, normalized_features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch
        
# def training(epoch, model, classifier, train_loader, optimizer, criterion):
#     "Training over an epoch"
#     metric_monitor = MetricMonitor()
#     model.eval()
#     classifier.train()
#     for batch_idx, (data,labels) in enumerate(train_loader):
#         if torch.cuda.is_available():
#             data,labels = data.cuda(), labels.cuda()
#         data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
#         with torch.no_grad():
#             features = model.encoder(data)
#         output = classifier(features.float())
#         loss = criterion(output, labels) 
#         accuracy = calculate_accuracy(output, labels)
#         metric_monitor.update("Loss", loss.item())
#         metric_monitor.update("Accuracy", accuracy)
#         data.detach()
#         labels.detach()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     print("[Epoch: {epoch:03d}] Train      | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
#     return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']


# def validation(epoch, model, classifier, valid_loader, criterion):
#     "Validation over an epoch"
#     metric_monitor = MetricMonitor()
#     model.eval()
#     classifier.eval()
#     with torch.no_grad():
#         for batch_idx, (data,labels) in enumerate(valid_loader):
#             if torch.cuda.is_available():
#                 data,labels = data.cuda(), labels.cuda()
#             data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
#             features = model.encoder(data)
#             output = classifier(features.float())
#             loss = criterion(output,labels) 
#             accuracy = calculate_accuracy(output, labels)
#             metric_monitor.update("Loss", loss.item())
#             metric_monitor.update("Accuracy", accuracy)
#             data.detach()
#             labels.detach()
#     print("[Epoch: {epoch:03d}] Validation | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
#     return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Accuracy']['avg']

class Temporal_Transformation:
    def __init__(self, shift_amount):
        self.shift_amount = shift_amount
        
    def __call__(self, x):
        
        # If shift_amount is zero, return the image as is
        if self.shift_amount == 0:
            return x

        # Downward shift
        if self.shift_amount > 0:
            top_padding = self.shift_amount
            bottom_padding = 0

        # Pad the image with zeros at the top and bottom
        padded_img = F.pad(x, (0, 0, top_padding, bottom_padding))
        
        # Slice the image to introduce the shift
        shifted_img = padded_img[:, :x.shape[1], :]

        return shifted_img
    


class CustomContrastiveDataset(Dataset):
    def __init__(self, tensor_data, tensor_labels, transform1=None, transform2=None):
        self.data = tensor_data
        self.labels = tensor_labels
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        lab = self.labels[index]
        
        x1 = self.transform1(x) if self.transform1 else x
        x2 = self.transform2(x) if self.transform2 else x

        return [[x1, x2], lab]


def main():
    
    num_epochs = 50
    use_early_stopping = True
    use_scheduler = True
    # head_type = 'mlp' # choose among 'mlp' and 'linear"
    # method = 'SimCLR' # choose among 'SimCLR' and 'SupCon'
    save_file = os.path.join('./results/', 'model.pth')
    if not os.path.isdir('./results/'):
         os.makedirs('./results/')

    first_aug = torch.load('/home/akapoor/Desktop/first_aug.pt')
    second_aug = torch.load('/home/akapoor/Desktop/second_aug.pt')
    labels = torch.load('/home/akapoor/Desktop/labels_tensor.pt')
    
    from torch.utils.data import DataLoader

    batch_size = 64  # or any other batch size you desire
    first_aug_dataset = TensorDataset(first_aug, labels)
    first_aug_dataloader = DataLoader(first_aug_dataset, batch_size=batch_size, shuffle=False)
    
    second_aug_dataset = TensorDataset(second_aug, labels)
    second_aug_dataloader = DataLoader(second_aug_dataset, batch_size=batch_size, shuffle=False)
    
    # Part 1
    encoder = Encoder()
    model = encoder
    # criterion = SupConLoss(temperature=temperature)
    if torch.cuda.is_available():
        model = model.cuda()
        # criterion = criterion.cuda()   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    
    # Ensure the model is on the desired device
    model = model.to('cpu')
    
    # Initialize lists to store data and labels
    model_rep_list_untrained = []
    labels_list_untrained = []
    
    # Iterate over the data
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_idx, ((data1, labels1), (data2, labels2)) in enumerate(zip(first_aug_dataloader, second_aug_dataloader)):
            # data = data[0]
            data = data1.to(torch.float32)
            features = model(data)
            model_rep_list_untrained.append(features)
            labels_list_untrained.append(labels1)
    
    # Convert lists to tensors
    model_rep_untrained = torch.cat(model_rep_list_untrained, dim=0)
    labels_arr_untrained = torch.cat(labels_list_untrained, dim=0)
    
    
        
    import umap
    import numpy as np
    reducer = umap.UMAP()
    untrained_rep_umap = reducer.fit_transform(model_rep_untrained.clone().detach().numpy())
    # Convert labels to numpy if they aren't already
    labels_numpy = labels_arr_untrained.clone().detach().numpy()
    
    # Get unique labels
    unique_labels = np.unique(labels_numpy)
    
    plt.figure()
    # Plot each group one by one with a label for the legend
    for label in unique_labels:
        mask = (labels_numpy == label)
        plt.scatter(untrained_rep_umap[mask, 0], untrained_rep_umap[mask, 1], label=str(label))
    
    # Show the legend
    plt.legend()
    
    # Show the plot
    plt.show()
    
    # Now let's train our model
    model = model.to('cuda')
    contrastive_loss, contrastive_lr = [], []
    negative_similarities_mega = []
    mean_cos_sim_epoch = []
    mean_ntxent_sim_epoch = []
    
    for epoch in range(1, num_epochs+1):
        model = model.to('cuda')
        loss, lr, negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch = pretraining(epoch, model, zip(first_aug_dataloader, second_aug_dataloader), optimizer)
        # loss, lr, negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch = pretraining(epoch, model, contrastive_loader, optimizer, criterion, method=method)
        if use_scheduler:
            scheduler.step()
        contrastive_loss.append(loss)
        contrastive_lr.append(lr)
        negative_similarities_mega.append(negative_similarities_for_epoch)
        mean_cos_sim_epoch.append(np.mean(mean_pos_cos_sim_for_epoch))
        mean_ntxent_sim_epoch.append(np.mean(mean_ntxent_positive_similarities_for_epoch))
        
    
    # Ensure the model is on the desired device
    model = model.to('cpu')
    
    # Initialize lists to store data and labels
    model_rep_list_trained = []
    labels_list_trained = []
    
    # Iterate over the data
    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch_idx, ((data1, labels1), (data2, labels2)) in enumerate(zip(first_aug_dataloader, second_aug_dataloader)):
            # data = data[0]
            data = data1.to(torch.float32)
            features = model(data)
            model_rep_list_trained.append(features)
            labels_list_trained.append(labels1)
    
    # Convert lists to tensors
    model_rep_trained = torch.cat(model_rep_list_trained, dim=0)
    labels_arr_trained = torch.cat(labels_list_trained, dim=0)
    
    
        
    import umap
    import numpy as np
    reducer = umap.UMAP()
    trained_rep_umap = reducer.fit_transform(model_rep_trained.clone().detach().numpy())
    # Convert labels to numpy if they aren't already
    labels_numpy = labels_arr_trained.clone().detach().numpy()
    
    # Get unique labels
    unique_labels = np.unique(labels_numpy)
    
    plt.figure()
    # Plot each group one by one with a label for the legend
    for label in unique_labels:
        mask = (labels_numpy == label)
        plt.scatter(trained_rep_umap[mask, 0], trained_rep_umap[mask, 1], label=str(label))
    
    # Show the legend
    plt.legend()
    # Show the plot
    plt.show()

    plt.figure()
    plt.plot(mean_cos_sim_epoch)
    plt.title("Mean Cosine Similarity of the Positive Sample Model Representations")
    plt.show()
    
    initial_sims = negative_similarities_mega[0][0]
    plt.figure()
    plt.hist(negative_similarities_mega[-1][0].clone().cpu().detach().numpy(), label = 'trained')
    plt.hist(initial_sims.clone().cpu().detach().numpy(), label = 'untrained')
    plt.legend()
    plt.show()
    
    # Extract and calculate the mean for the second sublist
    second_sublist_means = [sum(inner_list[1]) / len(inner_list[1]) for inner_list in negative_similarities_mega]
    means_list = [tensor.item() for tensor in second_sublist_means]

    # Plot
    plt.figure()
    plt.plot(means_list)
    plt.title('Mean NT-Xent Similarity for Negative Samples')
    plt.xlabel('Epoch')
    plt.ylabel('Similarity')
    plt.show()
    
    
    
    save_model(model, optimizer, num_epochs, save_file)
    plt.figure()
    plt.plot(range(1,len(contrastive_lr)+1),contrastive_lr, color='b', label = 'learning rate')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Learning Rate'), plt.show()
    
    plt.figure()
    plt.plot(range(1,len(contrastive_loss)+1),contrastive_loss, color='b', label = 'loss')
    plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Loss'), plt.show()
    

if __name__ == "__main__":
    main()
