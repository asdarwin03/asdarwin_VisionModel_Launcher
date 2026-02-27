import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch

inputsize = {
    "alexnet": 227,
    "resnet20": 32,
    "preactresnet110": 32,
    "densenetbc100": 32,
    "fractalnet40": 32,
}

step = 0
BATCH_TRACK_TIME=10

def getDefaultTransforms(model_name):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(inputsize[model_name]),
        transforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((inputsize[model_name], inputsize[model_name])),
    ])
    return train_transforms, test_transforms


def train(dataloader, model, loss_fn, optimizer, device, scheduler, cur_epoch, writer):
    global step
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
            print(f"avg loss: {avg_loss:>7f} [{current:>5d}/{size:>5d}]")

            """
            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/acc", avg_acc, step)
            """
            running_loss = 0.0
            running_correct = 0
            running_total = 0
        step += 1
    print(batches)
    total_loss /= batches
    total_correct /= size
    writer.add_scalar("train/loss", total_loss, cur_epoch)
    writer.add_scalar("train/acc", total_correct, cur_epoch)
    return total_correct, total_loss


def test(dataloader, model, loss_fn, device, cur_epoch, writer):
    size = len(dataloader.dataset)
    batches = len(dataloader)
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