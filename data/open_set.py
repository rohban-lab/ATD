import cv2
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


def food_loader(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(32,32))
    return img

def get_out_training_loaders(batch_size):

    trainset_out = torchvision.datasets.ImageFolder(root = 'data/food-101/images/', loader=food_loader, 
                                                    transform = transforms.Compose([transforms.ToPILImage(),
                                                                                    transforms.RandomChoice(
                                                                                        [transforms.RandomApply([transforms.RandomAffine(90, translate=(0.15, 0.15), scale=(0.85, 1), shear=None)], p=0.6),
                                                                                         transforms.RandomApply([transforms.RandomAffine(0, translate=None, scale=(0.5, 0.75), shear=30)], p=0.6),
                                                                                         transforms.RandomApply([transforms.AutoAugment()], p=0.9),]),
                                                                                    transforms.ToTensor(),  ]))
    trainloader_out = DataLoader(trainset_out, batch_size=batch_size, shuffle=True, num_workers=2)

    valset_out = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
    valloader_out = DataLoader(valset_out, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader_out, valloader_out

def bird_loader(path):
    path = path.split('/')
    if path[-1][0:2] == '._':
        path[-1] = path[-1][2:]
    path = '/'.join(path)
    img=cv2.imread(path)
    img=cv2.resize(img,(32,32))
    return img

def flower_loader(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(32,32))
    return img


def get_out_testing_datasets(out_names):

    out_datasets = []
    returned_out_names = []

    for name in out_names:

        if name == 'mnist':
            mnist = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                  transforms.Resize(32),
                                                                                                  transforms.Lambda(lambda x : x.repeat(3, 1, 1)),
                                                                                                  ]))
            returned_out_names.append(name)
            out_datasets.append(mnist)
        
        elif name == 'tiny_imagenet':
            tiny_imagenet = torchvision.datasets.ImageFolder(root = 'data/tiny-imagenet-200/test', transform=transforms.Compose([transforms.ToTensor(),
                                                                                                          transforms.Resize(32)]))
            
            returned_out_names.append(name)
            out_datasets.append(tiny_imagenet)
        
        elif name == 'places':
            places365 = torchvision.datasets.Places365(root = 'data/', split = 'val', small = True, download = False, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                          transforms.Resize(32)]))

            returned_out_names.append(name)
            out_datasets.append(places365)
        
        elif name == 'LSUN':
            LSUN = torchvision.datasets.ImageFolder(root = 'data/LSUN_resize/', transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(LSUN)

        elif name == 'iSUN':
            iSUN = torchvision.datasets.ImageFolder(root = 'data/iSUN/', transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(iSUN)
          
        elif name == 'birds': 
            birds = torchvision.datasets.ImageFolder(root = 'data/images/', loader=bird_loader, transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(birds)
        
        elif name == 'flowers':
            flowers = torchvision.datasets.ImageFolder(root = 'data/flowers/', loader=flower_loader, transform = transforms.ToTensor())

            returned_out_names.append(name)
            out_datasets.append(flowers)
          
        elif name == 'coil':
            coil_100 = torchvision.datasets.ImageFolder(root = 'data/coil/', transform=transforms.Compose([transforms.ToTensor(),
                                                                                                transforms.Resize(32)]))
            
            returned_out_names.append(name)
            out_datasets.append(coil_100)
        
        else:
          print(name, ' dataset is not implemented.')
    
    return returned_out_names, out_datasets