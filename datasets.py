import torchvision
import torchvision.transforms as transforms

class SimCLRTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def load_dataset(dataset_name, dst_path='./data', isTrain=False, img_size=(32, 32), isSimCLR=False, eval=False):
    transform = []
    if isTrain:
        if eval is True:
            transform = transforms.Compose([
                transforms.Resize((img_size[0], img_size[1])),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif isSimCLR is False:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((img_size[0], img_size[1])),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            # SimCLR transform
            transform = SimCLRTransform(transforms.Compose([
                transforms.RandomResizedCrop((img_size[0], img_size[1]), scale=(0.2, 1.0)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size[0], img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=dst_path, train=isTrain, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root=dst_path, train=isTrain, download=True, transform=transform)
        num_classes = 100

    # transform = transforms.Compose([..., Normalize(mean, std)])
    return dataset, num_classes

