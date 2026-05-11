from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch

BATCH_TRACK_TIME = 10
step = 0

def train(dataloader, model, loss_fn, optimizer, device, scheduler, cur_epoch, logger=None):
    global step
    writer = logger.writer
    size = len(dataloader.dataset)
    batches = len(dataloader)
    model.train()
    total_loss = 0.0
    total_correct = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        scheduler.step()

        correct = (pred.argmax(dim=1) == y).sum().item()
        total = y.size(0)

        running_loss += loss.item()
        total_loss += loss.item()
        running_correct += correct
        total_correct += correct
        running_total += total

        if (batch + 1) % BATCH_TRACK_TIME == 0:
            current = (batch + 1) * len(X)
            avg_loss = running_loss / BATCH_TRACK_TIME
            avg_acc = running_correct / running_total
            logger.print(f"avg loss: {avg_loss:>7f} [{current:>5d}/{size:>5d}]")

            """
            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/acc", avg_acc, step)
            """
            running_loss = 0.0
            running_correct = 0
            running_total = 0
        step += 1
    logger.print(batches)
    total_loss /= batches
    total_correct /= size
    writer.add_scalar("train/loss", total_loss, cur_epoch)
    writer.add_scalar("train/acc", total_correct, cur_epoch)
    return total_correct, total_loss


def pretrain(dataloader, model, loss_fn, optimizer, device, scheduler, cur_epoch, logger=None):
    global step
    writer = logger.writer
    size = len(dataloader.dataset)
    batches = len(dataloader)
    model.train()
    total_loss = 0.0
    total_correct = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    total_seen = 0

    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device) # for self-supervised learning
        optimizer.zero_grad()
        pred, y = model(X) # for self-supervised learning
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        scheduler.step()
        
        correct = (pred.argmax(dim=1) == y).sum().item()
        total = y.size(0)

        running_loss += loss.item()
        total_loss += loss.item()
        running_correct += correct
        total_correct += correct
        total_seen += total
        running_total += total

        if (batch + 1) % BATCH_TRACK_TIME == 0:
            current = (batch + 1) * len(X)
            avg_loss = running_loss / BATCH_TRACK_TIME
            avg_acc = running_correct / running_total
            logger.print(f"avg loss: {avg_loss:>7f} [{current:>5d}/{size:>5d}]")

            """
            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/acc", avg_acc, step)
            """
            running_loss = 0.0
            running_correct = 0
            running_total = 0
        step += 1
    logger.print(batches)
    total_loss /= batches
    total_correct /= total_seen
    writer.add_scalar("pretrain/loss", total_loss, cur_epoch)
    writer.add_scalar("pretrain/acc", total_correct, cur_epoch)
    return total_correct, total_loss

def SimCLR_train(dataloader, model, loss_fn, optimizer, device, scheduler, cur_epoch, logger=None):
    global step
    writer = logger.writer
    size = len(dataloader.dataset)
    batches = len(dataloader)
    model.train()
    total_loss = 0.0

    for batch, ((X1, X2), _) in enumerate(dataloader):
        X1 = X1.to(device) # for self-supervised learning
        X2 = X2.to(device) # for self-supervised learning
        optimizer.zero_grad()
        z_1= model(X1) # for self-supervised learning
        z_2 = model(X2) # for self-supervised learning
        loss = loss_fn(z_1, z_2)
        loss.backward()
        optimizer.step()

        scheduler.step()

        total_loss += loss.item()

        if (batch + 1) % BATCH_TRACK_TIME == 0:
            current = (batch + 1) * len(X1)
            avg_loss = total_loss / (batch + 1)
            logger.print(f"avg loss: {avg_loss:>7f} [{current:>5d}/{size:>5d}]")
        
    total_loss /= batches
    writer.add_scalar("train/loss", total_loss, cur_epoch)
    return total_loss


def test(dataloader, model, loss_fn, device, cur_epoch, logger=None):
    size = len(dataloader.dataset)
    batches = len(dataloader)
    writer = logger.writer
    model.eval()
    test_loss, correct = 0.0, 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)
        test_loss /= batches
        correct /= size

    writer.add_scalar("test/loss", test_loss, cur_epoch)
    writer.add_scalar("test/acc", correct, cur_epoch)
    return correct, test_loss



def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def knn_evaluate(encoder, train_loader, test_loader, device, k=5, num_classes=10):
    encoder.eval()
    train_features = []
    train_labels = []
    with torch.no_grad():
        for (X1, X2), y in train_loader:
            X1 = X1.to(device)
            X2 = X2.to(device)
            features = encoder(X1) # [batch_size, feature_dim]
            features2 = encoder(X2) # [batch_size, feature_dim]
            train_features.append(features.cpu())
            train_features.append(features2.cpu())
            train_labels.append(y.cpu())
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
    knn_one_hot = torch.nn.functional.one_hot(knn_labels, num_classes=num_classes).float()
    votes = knn_one_hot.sum(dim=1)
    pred_labels = votes.argmax(dim=1)
    accuracy = (pred_labels == test_labels).float().mean().item()
    
    return accuracy

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_1, z_2):
        batch_size = z_1.size(0)
        z_1 = nn.functional.normalize(z_1, dim=1)
        z_2 = nn.functional.normalize(z_2, dim=1)

        z = torch.cat([z_1, z_2], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        sim_matrix_exp = torch.exp(sim_matrix)

        mask = ~torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix_exp_masked = sim_matrix_exp*mask

        pos_sim = torch.exp(torch.sum(z_1 * z_2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        loss = -torch.log(pos_sim / sim_matrix_exp_masked.sum(dim=1))
        return loss.mean()