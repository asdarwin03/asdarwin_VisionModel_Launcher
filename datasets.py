import torchvision
import torchvision.transforms as transforms


def load_dataset(dataset_name, dst_path='./data', img_size=(32, 32)):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((img_size[0], img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if dataset_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=dst_path, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=dst_path, train=False, download=True, transform=test_transform)
        num_classes = 10

    elif dataset_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=dst_path, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root=dst_path, train=False, download=True, transform=test_transform)
        num_classes = 100

    # transform = transforms.Compose([..., Normalize(mean, std)])
    return trainset, testset, num_classes

