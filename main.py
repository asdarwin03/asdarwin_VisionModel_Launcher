from argparse import ArgumentParser
import torch
import json
import methods
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
    "AlexNet": 227,
    "ResNet20": 32,
    "PreActResNet110": 32,
    "DenseNetbc100": 32,
    "FractalNet40": 32,
    "FractalNet": 32,
    "VisionTransformer": 224,
    "MLPMixer": 224,
    "ConvMixer": 224,
    "RotNet": 227,
    "SimCLR": 32
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

def conf_launch(config):
    method = config['method']

    if method == "Supervised":
        launch_Supervised(config)
    elif method == "SimCLR":
        launch_SimCLR(config)
    elif method == "RotNet":
        launch_RotNet(config)
    elif method == "MoCo":
        launch_MoCo(config)
    else:
        print(f"method {method}is not specified.")
        exit(0)

def launch_Supervised(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = config['model']
    method_name = config['method']
    run_name = config.get('run_name')
    if run_name is None:
        run_name = f"{method_name}_{model_name}_{config['train_dataset_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    model_class = getattr(models, model_name)
    method_class = getattr(methods, method_name)

    # train dataset
    batch_size = config['train']['batch_size']
    train_dataset_name = config['train_dataset_name']
    train_dataset_path = config['train_dataset_path'] or None
    trainset, train_num_classes = datasets.load_dataset(dataset_name=train_dataset_name, dst_path=train_dataset_path,
                                                        isTrain=True, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # test dataset
    test_dataset_name = config['test_dataset_name']
    test_dataset_path = config['test_dataset_path'] or None
    testset, test_num_classes = datasets.load_dataset(dataset_name=test_dataset_name, dst_path=test_dataset_path,
                                                      isTrain=False, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # model / method
    model = model_class(dim_out=config['encoder_dim_out'])
    method = method_class(encoder=model, num_classes=train_num_classes)

    # log setup
    logdir = config['log_save_path']
    logdir = os.path.join(logdir, f"{method_name}_{model_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    log = logger.Logger(logdir)
    writer = log.writer
    log.print(device)
    log.print(config)
    log.print("tensorboard name: " + f"{run_name}")
    log.print(method)
    log.print(f"total num of classes: {test_num_classes}")


    method.to(device)
    train_config = config['train']
    optimizer, num_batches, num_epochs = None, None, None
    # optimizer config
    optimizer = optim.SGD(method.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'], momentum=train_config['momentum'])

    num_total_samples = len(trainloader.dataset)
    num_batches = len(trainloader)
    num_epochs = train_config['epochs']

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches * num_epochs * 2 / 4, num_batches * num_epochs * 3 / 4],
                            gamma=0.1)
    log.print(f"learning rate becomes gamma*lr when current epoch is {num_epochs * 2 / 4}, {num_epochs * 3 / 4}")

    for epoch in range(num_epochs):
        log.print(f"Epoch {epoch + 1}\n--------------------------")
        train_loss = utils.train(dataloader=trainloader, method=method, optimizer=optimizer,
                                            device=device, scheduler=scheduler, cur_epoch=epoch, logger=log)

        log.print(
            f"[epoch: {epoch + 1:>3d}/{num_epochs:>3d}] Train Avg loss: {train_loss:>8f}")
        test_loss = utils.test(dataloader=testloader, method=method, device=device,
                                         cur_epoch=epoch, logger=log)
        log.print(f"Test Result:\n Test Avg loss: {test_loss:>8f}\n")
    
    log.print("Train is now completed.")
    os.makedirs(config['model_save_path'], exist_ok=True)
    torch.save(method.state_dict(), os.path.join(config['model_save_path'], run_name + ".pth"))
    writer.close()
    log.print("Saved Model State to " + run_name + ".pth")






def launch_SimCLR(config):
    model_name = config['model']
    method_name = config['method']
    run_name = config.get('run_name')
    if run_name is None:
        run_name = f"{method_name}_{model_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    dim_out = config['encoder_dim_out']

    # SimCLR / Self-Supervised
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['train']['batch_size']

    model_class = getattr(models, model_name)
    method_class = getattr(methods, method_name)

    # train dataset
    train_dataset_name = config['train_dataset_name']
    train_dataset_path = config['train_dataset_path'] or None
    trainset, train_num_classes = datasets.load_dataset(dataset_name=train_dataset_name, dst_path=train_dataset_path,
                                                        isTrain=True, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]), isSimCLR=True)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # test dataset
    test_dataset_name = config['test_dataset_name']
    test_dataset_path = config['test_dataset_path'] or None
    testset, test_num_classes = datasets.load_dataset(dataset_name=test_dataset_name, dst_path=test_dataset_path,
                                                      isTrain=False, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # knn evaluation dataset for SimCLR
    if config.get('evaluation') == "kNN":
        # training dataset for kNN evaluation
        batch_size = config['train']['batch_size']
        knn_train_dataset_name = config['train_dataset_name']
        knn_train_dataset_path = config['train_dataset_path'] or None
        knn_trainset, knn_train_num_classes = datasets.load_dataset(dataset_name=knn_train_dataset_name, dst_path=knn_train_dataset_path,
                                                            isTrain=True, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]), eval=True)
        knn_trainloader = DataLoader(dataset=knn_trainset, batch_size=batch_size, shuffle=False, num_workers=0)

        # test dataset for kNN evaluation
        knn_test_dataset_name = config['test_dataset_name']
        knn_test_dataset_path = config['test_dataset_path'] or None
        knn_testset, knn_test_num_classes = datasets.load_dataset(dataset_name=knn_test_dataset_name, dst_path=knn_test_dataset_path,
                                                                  isTrain=False, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
        knn_testloader = DataLoader(dataset=knn_testset, batch_size=batch_size, shuffle=False, num_workers=0)


    model = model_class(dim_out=dim_out)
    method = method_class(encoder=model, z_dim=config.get('z_dim', 128), temperature=config['train'].get('temperature', 0.5))

    # logging setup
    logdir = config['log_save_path']
    logdir = os.path.join(logdir, f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}")

    log = logger.Logger(logdir)
    writer = log.writer

    log.print(device)
    log.print(config)
    log.print("tensorboard name: " + f"{run_name}")
    log.print(method)
    log.print(f"total num of classes: {test_num_classes}")

    method.to(device)


    train_config = config['train']
    # loss function config
    optimizer, num_batches, num_epochs = None, None, None

    # optimizer config
    opt = "SGD"
    init_learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    momentum = train_config['momentum']
    optimizer = optim.SGD(method.parameters(), lr=init_learning_rate, weight_decay=weight_decay, momentum=momentum)

    num_batches = len(trainloader)
    num_epochs = train_config['epochs']

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches * num_epochs * 2 / 4, num_batches * num_epochs * 3 / 4],
                            gamma=0.1)
    log.print(f"learning rate becomes gamma*lr when current epoch is {num_epochs * 2 / 4}, {num_epochs * 3 / 4}")

    for epoch in range(num_epochs):
        log.print(f"Epoch {epoch + 1}\n--------------------------")
        train_loss = utils.SimCLR_train(dataloader=trainloader, method=method, optimizer=optimizer,
                                            device=device, scheduler=scheduler, cur_epoch=epoch, logger=log)
        log.print(
            f"[epoch: {epoch + 1:>3d}/{num_epochs:>3d}] train Avg loss: {train_loss:>8f}")
        
        if config.get('evaluation') == "kNN":
            k = config.get('evaluation_config', {}).get('k', 5)
            # kNN evaluation
            test_acc = utils.knn_evaluate(
                encoder=method.encoder,
                train_loader=knn_trainloader,
                test_loader=knn_testloader,
                device=device,
                cur_epoch=epoch,
                k=k,
                num_classes=knn_test_num_classes,
                logger=log
            )
            log.print(f"Test Result:\nTest Accuracy(kNN evaluation): {(100 * test_acc):>0.1f}%\n")
        
        
    log.print("Train is now completed.")

    
    os.makedirs(config['model_save_path'], exist_ok=True)
    torch.save(method.state_dict(), os.path.join(config['model_save_path'], run_name + ".pth"))
    writer.close()
    log.print("Saved Model State to " + run_name + ".pth")


def launch_MoCo(config):
    model_name = config['model']
    method_name = config['method']
    run_name = config.get('run_name')
    if run_name is None:
        run_name = f"{method_name}_{model_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    dim_out = config['encoder_dim_out']

    # MoCo / Self-Supervised
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['train']['batch_size']

    model_class = getattr(models, model_name)
    method_class = getattr(methods, method_name)

    # train dataset
    train_dataset_name = config['train_dataset_name']
    train_dataset_path = config['train_dataset_path'] or None
    trainset, train_num_classes = datasets.load_dataset(dataset_name=train_dataset_name, dst_path=train_dataset_path,
                                                        isTrain=True, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]), isSimCLR=True)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True) # drop_last to make sure batch size is consistent for MoCo

    # test dataset
    test_dataset_name = config['test_dataset_name']
    test_dataset_path = config['test_dataset_path'] or None
    testset, test_num_classes = datasets.load_dataset(dataset_name=test_dataset_name, dst_path=test_dataset_path,
                                                      isTrain=False, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # knn evaluation dataset for SimCLR
    if config.get('evaluation') == "kNN":
        # training dataset for kNN evaluation
        batch_size = config['train']['batch_size']
        knn_train_dataset_name = config['train_dataset_name']
        knn_train_dataset_path = config['train_dataset_path'] or None
        knn_trainset, knn_train_num_classes = datasets.load_dataset(dataset_name=knn_train_dataset_name, dst_path=knn_train_dataset_path,
                                                            isTrain=True, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]), eval=True)
        knn_trainloader = DataLoader(dataset=knn_trainset, batch_size=batch_size, shuffle=False, num_workers=0)

        # test dataset for kNN evaluation
        knn_test_dataset_name = config['test_dataset_name']
        knn_test_dataset_path = config['test_dataset_path'] or None
        knn_testset, knn_test_num_classes = datasets.load_dataset(dataset_name=knn_test_dataset_name, dst_path=knn_test_dataset_path,
                                                                  isTrain=False, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
        knn_testloader = DataLoader(dataset=knn_testset, batch_size=batch_size, shuffle=False, num_workers=0)


    model = model_class(dim_out=dim_out)
    method = method_class(encoder=model, queue_size=config['train'].get('queue_size', 4096), temperature=config['train'].get('temperature', 0.5), momentum=config['train'].get('moco_momentum', 0.999))

    # logging setup
    logdir = config['log_save_path']
    logdir = os.path.join(logdir, f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}")

    log = logger.Logger(logdir)
    writer = log.writer

    log.print(device)
    log.print(config)
    log.print("tensorboard name: " + f"{run_name}")
    log.print(method)
    log.print(f"total num of classes: {test_num_classes}")

    method.to(device)


    train_config = config['train']
    # loss function config
    optimizer, num_batches, num_epochs = None, None, None

    # optimizer config
    opt = "SGD"
    init_learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    momentum = train_config['momentum']
    optimizer = optim.SGD(method.parameters(), lr=init_learning_rate, weight_decay=weight_decay, momentum=momentum)

    num_batches = len(trainloader)
    num_epochs = train_config['epochs']

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches * num_epochs * 2 / 4, num_batches * num_epochs * 3 / 4],
                            gamma=0.1)
    log.print(f"learning rate becomes gamma*lr when current epoch is {num_epochs * 2 / 4}, {num_epochs * 3 / 4}")

    for epoch in range(num_epochs):
        log.print(f"Epoch {epoch + 1}\n--------------------------")
        train_loss = utils.MoCo_train(dataloader=trainloader, method=method, optimizer=optimizer,
                                            device=device, scheduler=scheduler, cur_epoch=epoch, logger=log)
        log.print(
            f"[epoch: {epoch + 1:>3d}/{num_epochs:>3d}] train Avg loss: {train_loss:>8f}")
        
        if config.get('evaluation') == "kNN":
            k = config.get('evaluation_config', {}).get('k', 5)
            # kNN evaluation
            test_acc = utils.knn_evaluate(
                encoder=method.encoder,
                train_loader=knn_trainloader,
                test_loader=knn_testloader,
                device=device,
                cur_epoch=epoch,
                k=k,
                num_classes=knn_test_num_classes,
                logger=log
            )
            log.print(f"Test Result:\nTest Accuracy(kNN evaluation): {(100 * test_acc):>0.1f}%\n")
        
        
    log.print("Train is now completed.")

    
    os.makedirs(config['model_save_path'], exist_ok=True)
    torch.save(method.state_dict(), os.path.join(config['model_save_path'], run_name + ".pth"))
    writer.close()
    log.print("Saved Model State to " + run_name + ".pth")


def launch_RotNet(config):
    model_name = config['model']
    method_name = config['method']
    run_name = config.get('run_name')
    if run_name is None:
        run_name = f"{method_name}_{model_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    dim_out = config['encoder_dim_out']

    # RotNet / Self-Supervised
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['train']['batch_size']

    model_class = getattr(models, model_name)
    method_class = getattr(methods, method_name)

    # train dataset
    train_dataset_name = config['train_dataset_name']
    train_dataset_path = config['train_dataset_path'] or None
    trainset, train_num_classes = datasets.load_dataset(dataset_name=train_dataset_name, dst_path=train_dataset_path,
                                                        isTrain=True, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # test dataset
    test_dataset_name = config['test_dataset_name']
    test_dataset_path = config['test_dataset_path'] or None
    testset, test_num_classes = datasets.load_dataset(dataset_name=test_dataset_name, dst_path=test_dataset_path,
                                                      isTrain=False, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model_class(dim_out=dim_out)
    method = method_class(encoder=model)

    # logging setup
    logdir = config['log_save_path']
    logdir = os.path.join(logdir, f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}")

    log = logger.Logger(logdir)
    writer = log.writer

    log.print(device)
    log.print(config)
    log.print("tensorboard name: " + f"{run_name}")
    log.print(method)
    log.print(f"total num of classes: {test_num_classes}")

    method.to(device)


    train_config = config['train']
    # loss function config
    optimizer, num_batches, num_epochs = None, None, None

    # optimizer config
    opt = "SGD"
    init_learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    momentum = train_config['momentum']
    optimizer = optim.SGD(method.parameters(), lr=init_learning_rate, weight_decay=weight_decay, momentum=momentum)

    num_batches = len(trainloader)
    num_epochs = train_config['epochs']

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches * num_epochs * 2 / 4, num_batches * num_epochs * 3 / 4],
                            gamma=0.1)
    log.print(f"learning rate becomes gamma*lr when current epoch is {num_epochs * 2 / 4}, {num_epochs * 3 / 4}")

    for epoch in range(num_epochs):
        log.print(f"Epoch {epoch + 1}\n--------------------------")
        train_loss = utils.RotNet_train(dataloader=trainloader, method=method, optimizer=optimizer,
                                            device=device, scheduler=scheduler, cur_epoch=epoch, logger=log)

        log.print(
            f"[epoch: {epoch + 1:>3d}/{num_epochs:>3d}] train Avg loss: {train_loss:>8f}")
        
        # kNN evaluation
        if config.get('evaluation') == "kNN":
            k = config.get('evaluation_config', {}).get('k', 5)
            test_acc = utils.knn_evaluate(
                encoder=method.encoder,
                train_loader=trainloader,
                test_loader=testloader,
                device=device,
                cur_epoch=epoch,
                k=k,
                num_classes=test_num_classes,
                logger=log
            )
            log.print(f"Test Result:\nTest Accuracy: {(100 * test_acc):>0.1f}%\n")

    log.print("Train is now completed.")
    
    os.makedirs(config['model_save_path'], exist_ok=True)
    torch.save(method.state_dict(), os.path.join(config['model_save_path'], run_name + ".pth"))
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

    parser.add_argument("--config", "-c", type=str, default=None)  # file_based
    # test_only?
    args = parser.parse_args()

    if args.config is None:
        print("error")
        exit(0)

    with open(args.config, "r") as f:
        config = json.load(f)
        print(config)
        conf_launch(config)
        exit(0)

    if args.model == None or args.dataset == None:
        print("model, and dataset is necessary field.")
        exit(0)
    # launch(args)
