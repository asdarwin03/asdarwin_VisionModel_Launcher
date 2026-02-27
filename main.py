from argparse import ArgumentParser
import torch
import models, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import time
import os
import utils

# python main.py --model resnet20 --dataset cifar10 [[--method rotnet]] --logdir logs/{dataset}_{model_name} --adaptive_LR [50%, 75%, ...] or "just basic" --epoch () --batch_size
# hyper_parameters:
# epoch, batch_size, adaptive_LR, WEIGHT_DECAY, MOMENTUM(0.9 in general)
#


# python main.py 만 치면
# python main.py --model resnet20 --dataset cifar10 [[--method rotnet]] --logdir logs/{dataset}_{model_name} --adaptive_LR [50%, 75%, ...] or "just basic" --epoch () --batch_size
# python main.py --config [file_name]

# python main.py

BATCH_TRACK_TIME = 10

def launch(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(args)
    # args.model = "resnet20"
    # args.dataset = "cifar10"

    model_class = getattr(models, args.model)
    dataset_class = getattr(datasets, args.dataset)
    train_transforms, test_transforms = utils.getDefaultTransforms(model_name=args.model)

    custom_mean, custom_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # modify if you want. if you don't want to use custom mean/std, set to "auto".

    dataset = dataset_class(train_transform=train_transforms, test_transform=test_transforms, mean=custom_mean, std=custom_std)
    trainloader = DataLoader(dataset=dataset.getTrainSet(), batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=dataset.getTestSet(), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = model_class(num_classes=dataset.getNumClasses())
    run_name = f"{args.model}_{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}"

    if args.logdir == "":
        logdir = os.path.join("C:/logs", run_name)
    else:
        logdir = args.logdir

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)
    print("tensorboard name: " + f"{run_name}")

    print(model)
    print(f"total num of classes: {dataset.getNumClasses()}")

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    num_total_samples = len(trainloader.dataset)
    num_batches = len(trainloader)
    num_epochs = args.epoch

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches*num_epochs*2/4, num_batches*num_epochs*3/4], gamma=0.1)
    print(num_batches*num_epochs*2/4, num_batches*num_epochs*3/4)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n--------------------------")
        train_acc, train_loss = utils.train(dataloader=trainloader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, scheduler=scheduler, cur_epoch=epoch, writer=writer)

        print(f"[epoch: {epoch + 1:>3d}/{num_epochs:>3d}] Train Accuracy: {(100 * train_acc):>0.1f}%, Train Avg loss: {train_loss:>8f}")
        test_acc, test_loss = utils.test(dataloader=testloader, model=model, loss_fn=loss_fn, device=device, cur_epoch=epoch, writer=writer)
        print(f"Test Result:\nTest Accuracy: {(100 * test_acc):>0.1f}%, Test Avg loss: {test_loss:>8f}\n")
    print("Train is now completed.")
    torch.save(model.state_dict(), run_name + ".pth")
    writer.close()
    print("Saved Model State to " + run_name + ".pth")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--batch_size", "-bs", type=int, default=256)
    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument("--epoch", "-ep", type=int, default=20)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--logdir", type=str, default="")
    # test_only?

    args = parser.parse_args()
    print(args)
    if args.model == None or args.dataset == None:
        print("model, and dataset is necessary field.")
        exit(0)
    launch(args)
