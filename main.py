from argparse import ArgumentParser
import torch
import json
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
    run_mode = config['run_mode']
    train_type = config['train_type']

    if run_mode == "TrainAndTest" and train_type == "Supervised":
        launch_TrainAndTest_Supervised(config)
    elif run_mode == "PreTrainAndFineTuneAndTest" and train_type == "Self-Supervised":
        launch_PreTrainAndFineTuneAndTest_SelfSupervised(config)
    elif run_mode == "SimCLR":
        launch_SimCLR(config)

def launch_TrainAndTest_Supervised(config):
    run_name = config['run_name']
    run_mode = config['run_mode']
    model_name = config['model']
    train_type = config['train_type']

    if run_mode != "TrainAndTest" or train_type != "Supervised":
        exit(0)

    # TrainAndTest / Supervised
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['train']['batch_size']
    model_class = getattr(models, model_name)

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

    # model (config)
    model = model_class(num_classes=train_num_classes, net_config=config['train']['net_config'])


    logdir = config['log_save_path']
    logdir = os.path.join(logdir, f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}")

    log = logger.Logger(logdir)
    writer = log.writer

    log.print(device)
    log.print(config)
    log.print("tensorboard name: " + f"{run_name}")
    log.print(model)
    log.print(f"total num of classes: {test_num_classes}")

    model.to(device)

    train_config = config['train']

    # loss function config
    loss_func = train_config['loss_function']
    loss_fn, optimizer, num_batches, num_epochs = None, None, None, None
    if loss_func == "CrossEntropyLoss":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_func == "L1Loss":
        loss_fn = nn.L1Loss()
    elif loss_func == "MSELoss":
        loss_fn = nn.MSELoss()

    # optimizer config
    opt = train_config['optimizer']
    if opt == "SGD":
        init_learning_rate = train_config['learning_rate']
        weight_decay = train_config['weight_decay']
        momentum = train_config['momentum']
        optimizer = optim.SGD(model.parameters(), lr=init_learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif opt == "Adam":
        init_learning_rate = train_config['learning_rate']
        optimizer = optim.Adam(model.parameters(), lr=init_learning_rate)

    num_total_samples = len(trainloader.dataset)
    num_batches = len(trainloader)
    num_epochs = train_config['epochs']

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches * num_epochs * 2 / 4, num_batches * num_epochs * 3 / 4],
                            gamma=0.1)
    log.print(f"learning rate becomes gamma*lr when current epoch is {num_epochs * 2 / 4}, {num_epochs * 3 / 4}")

    for epoch in range(num_epochs):
        log.print(f"Epoch {epoch + 1}\n--------------------------")
        train_acc, train_loss = utils.train(dataloader=trainloader, model=model, loss_fn=loss_fn, optimizer=optimizer,
                                            device=device, scheduler=scheduler, cur_epoch=epoch, logger=log)

        log.print(
            f"[epoch: {epoch + 1:>3d}/{num_epochs:>3d}] Train Accuracy: {(100 * train_acc):>0.1f}%, Train Avg loss: {train_loss:>8f}")
        test_acc, test_loss = utils.test(dataloader=testloader, model=model, loss_fn=loss_fn, device=device,
                                         cur_epoch=epoch, logger=log)
        log.print(f"Test Result:\nTest Accuracy: {(100 * test_acc):>0.1f}%, Test Avg loss: {test_loss:>8f}\n")
    log.print("Train is now completed.")
    torch.save(model.state_dict(), os.path.join(config['model_save_path'], run_name + ".pth"))
    writer.close()
    log.print("Saved Model State to " + run_name + ".pth")



def launch_PreTrainAndFineTuneAndTest_SelfSupervised(config):
    run_name = config['run_name']
    run_mode = config['run_mode']
    model_name = config['model']
    train_type = config['train_type']

    if run_mode != "PreTrainAndFineTuneAndTest" or train_type != "Self-Supervised":
        exit(0)

    # PreTrainAndFineTuneAndTest / Self-Supervised
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['pretrain']['batch_size']
    model_class = getattr(models, model_name)

    # pretrain dataset
    pretrain_dataset_name = config['pretrain_dataset_name']
    pretrain_dataset_path = config['pretrain_dataset_path'] or None
    pretrainset, pretrain_num_classes = datasets.load_dataset(dataset_name=pretrain_dataset_name, dst_path=pretrain_dataset_path,
                                                        isTrain=True, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
    pretrainloader = DataLoader(dataset=pretrainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # finetune dataset
    finetune_dataset_name = config['finetune_dataset_name']
    finetune_dataset_path = config['finetune_dataset_path'] or None
    finetuneset, finetune_num_classes = datasets.load_dataset(dataset_name=finetune_dataset_name, dst_path=finetune_dataset_path,
                                                        isTrain=True, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
    finetuneloader = DataLoader(dataset=finetuneset, batch_size=batch_size, shuffle=True, num_workers=0)

    # test dataset
    test_dataset_name = config['test_dataset_name']
    test_dataset_path = config['test_dataset_path'] or None
    testset, test_num_classes = datasets.load_dataset(dataset_name=test_dataset_name, dst_path=test_dataset_path,
                                                      isTrain=False, img_size=(IMG_SIZE[model_name], IMG_SIZE[model_name]))
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # model (config)
    encoder_model = config['pretrain']['net_config']['encoder_model']
    if encoder_model == "AlexNet":
        encoder_class = getattr(models, encoder_model)
        encoder = encoder_class(num_classes=pretrain_num_classes, net_config=config['pretrain']['net_config']['encoder_config'])
    else:
        exit(0)
    model = model_class(net_config=config['pretrain']['net_config'], encoder=encoder)

    # logging setup
    logdir = config['log_save_path']
    logdir = os.path.join(logdir, f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}")

    log = logger.Logger(logdir)
    writer = log.writer

    log.print(device)
    log.print(config)
    log.print("tensorboard name: " + f"{run_name}")
    log.print(model)
    log.print(f"total num of classes: {test_num_classes}")

    model.to(device)


    train_config = config['pretrain']
    # loss function config
    loss_func = train_config['loss_function']
    loss_fn, optimizer, num_batches, num_epochs = None, None, None, None
    if loss_func == "CrossEntropyLoss":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_func == "L1Loss":
        loss_fn = nn.L1Loss()
    elif loss_func == "MSELoss":
        loss_fn = nn.MSELoss()

    # optimizer config
    opt = train_config['optimizer']
    if opt == "SGD":
        init_learning_rate = train_config['learning_rate']
        weight_decay = train_config['weight_decay']
        momentum = train_config['momentum']
        optimizer = optim.SGD(model.parameters(), lr=init_learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif opt == "Adam":
        init_learning_rate = train_config['learning_rate']
        optimizer = optim.Adam(model.parameters(), lr=init_learning_rate)

    num_total_samples = len(pretrainloader.dataset)
    num_batches = len(pretrainloader)
    num_epochs = train_config['epochs']

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches * num_epochs * 2 / 4, num_batches * num_epochs * 3 / 4],
                            gamma=0.1)
    log.print(f"learning rate becomes gamma*lr when current epoch is {num_epochs * 2 / 4}, {num_epochs * 3 / 4}")

    for epoch in range(num_epochs):
        log.print(f"Epoch {epoch + 1}\n--------------------------")
        train_acc, train_loss = utils.pretrain(dataloader=pretrainloader, model=model, loss_fn=loss_fn, optimizer=optimizer,
                                            device=device, scheduler=scheduler, cur_epoch=epoch, logger=log)

        log.print(
            f"[epoch: {epoch + 1:>3d}/{num_epochs:>3d}] Pretrain Accuracy: {(100 * train_acc):>0.1f}%, Pretrain Avg loss: {train_loss:>8f}")
        #test_acc, test_loss = utils.test(dataloader=testloader, model=model, loss_fn=loss_fn, device=device,
        #                                 cur_epoch=epoch, logger=log)
        #log.print(f"Test Result:\nTest Accuracy: {(100 * test_acc):>0.1f}%, Test Avg loss: {test_loss:>8f}\n")
    log.print("PreTrain is now completed.")

    # now finetune and test
    # ...

    model.setmode("finetune")
    train_config = config['finetune']
    # loss function config
    loss_func = train_config['loss_function']
    loss_fn, optimizer, num_batches, num_epochs = None, None, None, None
    if loss_func == "CrossEntropyLoss":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_func == "L1Loss":
        loss_fn = nn.L1Loss()
    elif loss_func == "MSELoss":
        loss_fn = nn.MSELoss()

    # optimizer config
    opt = train_config['optimizer']
    if opt == "SGD":
        init_learning_rate = train_config['learning_rate']
        weight_decay = train_config['weight_decay']
        momentum = train_config['momentum']
        optimizer = optim.SGD(model.parameters(), lr=init_learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif opt == "Adam":
        init_learning_rate = train_config['learning_rate']
        optimizer = optim.Adam(model.parameters(), lr=init_learning_rate)

    num_total_samples = len(finetuneloader.dataset)
    num_batches = len(finetuneloader)
    num_epochs = train_config['epochs']

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches * num_epochs * 2 / 4, num_batches * num_epochs * 3 / 4],
                            gamma=0.1)
    log.print(f"learning rate becomes gamma*lr when current epoch is {num_epochs * 2 / 4}, {num_epochs * 3 / 4}")

    for epoch in range(num_epochs):
        log.print(f"Epoch {epoch + 1}\n--------------------------")
        train_acc, train_loss = utils.train(dataloader=finetuneloader, model=model, loss_fn=loss_fn, optimizer=optimizer,
                                            device=device, scheduler=scheduler, cur_epoch=epoch, logger=log)

        log.print(
            f"[epoch: {epoch + 1:>3d}/{num_epochs:>3d}] Finetune Accuracy: {(100 * train_acc):>0.1f}%, Finetune Avg loss: {train_loss:>8f}")
        test_acc, test_loss = utils.test(dataloader=testloader, model=model, loss_fn=loss_fn, device=device,
                                         cur_epoch=epoch, logger=log)
        log.print(f"Test Result:\nTest Accuracy: {(100 * test_acc):>0.1f}%, Test Avg loss: {test_loss:>8f}\n")
    log.print("Finetune is now completed.")


    torch.save(model.state_dict(), os.path.join(config['model_save_path'], run_name + ".pth"))
    writer.close()
    log.print("Saved Model State to " + run_name + ".pth")


def launch_SimCLR(config):
    run_name = config['run_name']
    run_mode = config['run_mode']
    model_name = config['model']
    train_type = config['train_type']

    if run_mode != "SimCLR":
        exit(0)

    # SimCLR / Self-Supervised
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['train']['batch_size']
    model_class = getattr(models, model_name)

    # pretrain dataset
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

    # model (config)
    encoder_model = config['train']['net_config']['encoder_model']
    if encoder_model == "AlexNet" or encoder_model == "ResNet20":
        encoder_class = getattr(models, encoder_model)
        encoder = encoder_class(num_classes=config['train']['net_config']['encoder_config']['num_classes'], net_config=config['train']['net_config']['encoder_config'])
    else:
        exit(0)
    model = model_class(net_config=config['train']['net_config'], encoder=encoder, feature_dim=config['train']['net_config']['feature_dim'])

    # logging setup
    logdir = config['log_save_path']
    logdir = os.path.join(logdir, f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}")

    log = logger.Logger(logdir)
    writer = log.writer

    log.print(device)
    log.print(config)
    log.print("tensorboard name: " + f"{run_name}")
    log.print(model)
    log.print(f"total num of classes: {test_num_classes}")

    model.to(device)


    train_config = config['train']
    # loss function config
    loss_func = train_config['loss_function']
    loss_fn, optimizer, num_batches, num_epochs = None, None, None, None
    if loss_func == "CrossEntropyLoss":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_func == "L1Loss":
        loss_fn = nn.L1Loss()
    elif loss_func == "MSELoss":
        loss_fn = nn.MSELoss()
    elif loss_func == "NT-XentLoss":
        loss_fn = utils.NTXentLoss(temperature=0.5)

    # optimizer config
    opt = train_config['optimizer']
    if opt == "SGD":
        init_learning_rate = train_config['learning_rate']
        weight_decay = train_config['weight_decay']
        momentum = train_config['momentum']
        optimizer = optim.SGD(model.parameters(), lr=init_learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif opt == "Adam":
        init_learning_rate = train_config['learning_rate']
        optimizer = optim.Adam(model.parameters(), lr=init_learning_rate)

    num_total_samples = len(trainloader.dataset)
    num_batches = len(trainloader)
    num_epochs = train_config['epochs']

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches * num_epochs * 2 / 4, num_batches * num_epochs * 3 / 4],
                            gamma=0.1)
    log.print(f"learning rate becomes gamma*lr when current epoch is {num_epochs * 2 / 4}, {num_epochs * 3 / 4}")

    for epoch in range(num_epochs):
        log.print(f"Epoch {epoch + 1}\n--------------------------")
        train_loss = utils.SimCLR_train(dataloader=trainloader, model=model, loss_fn=loss_fn, optimizer=optimizer,
                                            device=device, scheduler=scheduler, cur_epoch=epoch, logger=log)

        log.print(
            f"[epoch: {epoch + 1:>3d}/{num_epochs:>3d}] train Avg loss: {train_loss:>8f}")
    log.print("Train is now completed.")

    # now evaluation
    # ...

    num_total_samples = len(testloader.dataset)
    num_batches = len(testloader)

    # adaptive learning rate is automatically applied. (50%, 75%)
    scheduler = MultiStepLR(optimizer, milestones=[num_batches * num_epochs * 2 / 4, num_batches * num_epochs * 3 / 4],
                            gamma=0.1)
    log.print(f"learning rate becomes gamma*lr when current epoch is {num_epochs * 2 / 4}, {num_epochs * 3 / 4}")

    test_acc = utils.knn_evaluate(encoder=model.encoder, train_loader=trainloader, test_loader=testloader, device=device, k=5, num_classes=test_num_classes)
    log.print(f"Test Result:\nTest Accuracy: {(100 * test_acc):>0.1f}%\n")

    torch.save(model.state_dict(), os.path.join(config['model_save_path'], run_name + ".pth"))
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
