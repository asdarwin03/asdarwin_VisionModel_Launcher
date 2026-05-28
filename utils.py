from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
BATCH_TRACK_TIME = 10
step = 0

def train(dataloader, method, optimizer, device, scheduler, cur_epoch, logger=None):
    global step
    writer = logger.writer
    size = len(dataloader.dataset)
    batches = len(dataloader)
    method.train()
    total_loss = 0.0
    running_loss = 0.0
    running_total = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = method(X, y) # return loss directly
        loss.backward()
        optimizer.step()
        scheduler.step()

        total = y.size(0)
        running_loss += loss.item()
        total_loss += loss.item()
        running_total += total

        if (batch + 1) % BATCH_TRACK_TIME == 0:
            current = (batch + 1) * len(X)
            avg_loss = running_loss / BATCH_TRACK_TIME
            logger.print(f"avg loss: {avg_loss:>7f} [{current:>5d}/{size:>5d}]")
            running_loss = 0.0
            running_total = 0
        step += 1
    logger.print(batches)
    total_loss /= batches
    writer.add_scalar("train/loss", total_loss, cur_epoch)
    return total_loss

def SimCLR_train(dataloader, method, optimizer, device, scheduler, cur_epoch, logger=None):
    global step
    writer = logger.writer
    size = len(dataloader.dataset)
    batches = len(dataloader)

    method.train()
    total_loss = 0.0

    for batch, ((x1, x2), _) in enumerate(dataloader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        optimizer.zero_grad()
        loss = method(x1, x2) # for self-supervised learning
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch + 1) % BATCH_TRACK_TIME == 0:
            current = (batch + 1) * len(x1)
            avg_loss = total_loss / (batch + 1)
            logger.print(f"avg loss: {avg_loss:>7f} [{current:>5d}/{size:>5d}]")
        
    total_loss /= batches
    writer.add_scalar("train/loss", total_loss, cur_epoch)
    return total_loss

def MoCo_train(dataloader, method, optimizer, device, scheduler, cur_epoch, logger=None):
    global step
    writer = logger.writer
    size = len(dataloader.dataset)
    batches = len(dataloader)

    method.train()
    total_loss = 0.0

    for batch, ((x_q, x_k), _) in enumerate(dataloader):
        x_q = x_q.to(device)
        x_k = x_k.to(device)
        optimizer.zero_grad()
        loss = method(x_q, x_k) # for self-supervised learning
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch + 1) % BATCH_TRACK_TIME == 0:
            current = (batch + 1) * len(x_q)
            avg_loss = total_loss / (batch + 1)
            logger.print(f"avg loss: {avg_loss:>7f} [{current:>5d}/{size:>5d}]")
        
    total_loss /= batches
    writer.add_scalar("train/loss", total_loss, cur_epoch)
    return total_loss


def RotNet_train(dataloader, method, optimizer, device, scheduler, cur_epoch, logger=None):
    global step
    writer = logger.writer
    size = len(dataloader.dataset)
    batches = len(dataloader)

    method.train()
    total_loss = 0.0

    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = method((x, y)) # for self-supervised learning
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        
    total_loss /= batches
    writer.add_scalar("train/loss", total_loss, cur_epoch)
    return total_loss


def test(dataloader, method, device, cur_epoch, logger=None):
    size = len(dataloader.dataset)
    batches = len(dataloader)
    writer = logger.writer
    method.eval()
    test_loss = 0.0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            loss = method(X, y) # return loss directly
            test_loss += loss.item()
            total += y.size(0)
        test_loss /= batches

    writer.add_scalar("test/loss", test_loss, cur_epoch)
    return test_loss



def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def knn_evaluate(encoder, train_loader, test_loader, device, cur_epoch, k=5, num_classes=10, logger=None):
    encoder.eval()
    train_features = []
    train_labels = []
    with torch.no_grad():
        for X, y in train_loader:
            X = X.to(device)
            features = encoder(X) # [batch_size, feature_dim]
            train_features.append(features.cpu())
            train_labels.append(y.cpu())
    train_features = torch.cat(train_features)
    train_labels = torch.cat(train_labels)

    test_features = []
    test_labels = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            features = encoder(X) # [batch_size, feature_dim]
            test_features.append(features.cpu())
            test_labels.append(y.cpu())
    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)

    # get distances between test features and train features
    dists = torch.cdist(test_features, train_features)
    knn_indices = dists.topk(k, largest=False).indices
    knn_labels = train_labels[knn_indices]
    
    # get majority vote
    knn_one_hot = F.one_hot(knn_labels, num_classes=num_classes).float()
    votes = knn_one_hot.sum(dim=1)
    pred_labels = votes.argmax(dim=1)
    accuracy = (pred_labels == test_labels).float().mean().item()
    if logger:
        logger.writer.add_scalar("test/knn_accuracy", accuracy, cur_epoch)
    return accuracy

