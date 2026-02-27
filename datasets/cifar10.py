import torchvision
import torchvision.transforms as transforms
import numpy as np

class cifar10:
    def __init__(self, train_transform, test_transform, mean="auto", std="auto"):
        self.mean, self.std = mean, std

        if mean == "auto" or std == "auto":
            print("mean/std is not set, dataset mean/std calculating.")
            auto_transform = transforms.Compose([transforms.ToTensor(),])
            temp_trainset = torchvision.datasets.CIFAR10(
                root='./data',
                train=True,
                download=True,
                transform=auto_transform
            )
            _mean, _std = self.calculateNorm(temp_trainset)
            print(f"mean: {_mean}, std:{_std}")
            if mean == "auto":
                self.mean = _mean
                print("mean calculated and applied")
            if std == "auto":
                self.std = _std
                print("std calculated and applied")

        self.train_transform = train_transform
        self.train_transform.transforms.append(transforms.ToTensor())
        self.train_transform.transforms.append(transforms.Normalize(self.mean, self.std))

        self.test_transform = test_transform
        self.test_transform.transforms.append(transforms.ToTensor())
        self.test_transform.transforms.append(transforms.Normalize(self.mean, self.std))

        self.trainset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=self.train_transform
        )
        self.testset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=self.test_transform
        )
        self.num_classes = len(self.testset.classes)

    def getTrainSet(self):
        return self.trainset

    def getTestSet(self):
        return self.testset

    def getNumClasses(self):
        return self.num_classes

    def calculateNorm(self, dataset):
        # https://teddylee777.github.io/pytorch/torchvision-transform/
        mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])
        mean_r = mean_[:, 0].mean()
        mean_g = mean_[:, 1].mean()
        mean_b = mean_[:, 2].mean()

        # dataset의 axis=1, 2에 대한 표준편차 산출
        std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])
        # r, g, b 채널에 대한 각각의 표준편차 산출
        std_r = std_[:, 0].mean()
        std_g = std_[:, 1].mean()
        std_b = std_[:, 2].mean()

        return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)