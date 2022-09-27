import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms

def get_in_training_loaders(in_dataset, batch_size):

    if in_dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'TI':
        dataset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/train', transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, valloader


def get_in_testing_loader(in_dataset, batch_size):

    if in_dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'cifar100':
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transforms.ToTensor())

    elif in_dataset == 'TI':
        testset = torchvision.datasets.ImageFolder(root = './data/tiny-imagenet-200/val', transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader