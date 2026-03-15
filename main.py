from argparse import ArgumentParser
import torch
import models, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import os
import utils
import logger

# python main.py --model resnet20 --dataset cifar10 [[--method rotnet]] --logdir logs/{dataset}_{model_name} --adaptive_LR [50%, 75%, ...] or "just basic" --epoch () --batch_size
# hyper_parameters:
# epoch, batch_size, adaptive_LR, WEIGHT_DECAY, MOMENTUM(0.9 in general)
#


# python main.py 만 치면
# python main.py --model resnet20 --dataset cifar10 [[--method rotnet]] --logdir logs/{dataset}_{model_name} --adaptive_LR [50%, 75%, ...] or "just basic" --epoch () --batch_size
# python main.py --config [file_name]

# python main.py

IMG_SIZE = {
    "alexnet": 227,
    "resnet20": 32,
    "preactresnet110": 32,
    "densenetbc100": 32,
    "fractalnet40": 32,
    "fractalnet": 32,
    "visiontransformer": 224,
    "mlpmixer": 224,
    "convmixer": 224,
}


def launch(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.model = "resnet20"
    # args.dataset = "cifar10"

    model_class = getattr(models, args.model)
    trainset, testset, num_classes = datasets.load_dataset(name=args.dataset, img_size=(IMG_SIZE[args.model], IMG_SIZE[args.model]))

    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = model_class(num_classes=num_classes)
    run_name = f"{args.model}_{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}"

    if args.logdir == "":
        logdir = os.path.join("C:/logs", run_name)
    else:
        logdir = args.logdir

    log = logger.Logger(logdir)
    writer = log.writer

    log.print(device)
    log.print(args)
    log.print("tensorboard name: " + f"{run_name}")
    log.print(model)
    log.print(f"total num of classes: {num_classes}")

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    num_total_samples = len(trainloader.dataset)
    num_batches = len(trainloader)
    num_epochs = args.epoch

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches*num_epochs*2/4, num_batches*num_epochs*3/4], gamma=0.1)
    log.print(f"learning rate becomes gamma*lr when current epoch is {num_epochs*2/4}, {num_epochs*3/4}")

    for epoch in range(num_epochs):
        log.print(f"Epoch {epoch + 1}\n--------------------------")
        train_acc, train_loss = utils.train(dataloader=trainloader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, scheduler=scheduler, cur_epoch=epoch, logger=log)

        log.print(f"[epoch: {epoch + 1:>3d}/{num_epochs:>3d}] Train Accuracy: {(100 * train_acc):>0.1f}%, Train Avg loss: {train_loss:>8f}")
        test_acc, test_loss = utils.test(dataloader=testloader, model=model, loss_fn=loss_fn, device=device, cur_epoch=epoch, logger=log)
        log.print(f"Test Result:\nTest Accuracy: {(100 * test_acc):>0.1f}%, Test Avg loss: {test_loss:>8f}\n")
    log.print("Train is now completed.")
    torch.save(model.state_dict(), run_name + ".pth")
    writer.close()
    log.print("Saved Model State to " + run_name + ".pth")

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
    if args.model == None or args.dataset == None:
        print("model, and dataset is necessary field.")
        exit(0)
    launch(args)
