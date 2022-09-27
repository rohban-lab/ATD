#utils.py

import torch
from torchvision import transforms

from robustbench import load_model
import numpy as np
import random

from models.preact_resnet import ti_preact_resnet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


#normilizer model
class normalizer():
    def __init__(self, model):
        self.model = model
        self.transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    def __call__(self, x):
        out = self.model(self.transform(x))
        return out


def get_feature_extractor_model(training_type, in_dataset):

    if training_type == 'adv':
    
        if in_dataset == 'cifar10':
            model = load_model(model_name='Rade2021Helper_R18_extra', dataset='cifar10', threat_model='Linf').to(device)
            model.logits = torch.nn.Sequential()
            model.eval()

        elif in_dataset == 'cifar100':
            model = load_model(model_name='Rade2021Helper_R18_ddpm', dataset='cifar100', threat_model='Linf').to(device)
            model.logits = torch.nn.Sequential()
            model.eval()
        
        
        elif in_dataset == 'TI':
            ckpt = torch.load("models/weights-best-TI.pt")['model_state_dict']
            
            model = ti_preact_resnet('preact-resnet18', num_classes=200).to(device)    
            model = torch.nn.Sequential(model)
            model = torch.nn.DataParallel(model).to(device)
            
            model.load_state_dict(ckpt)
            model.module[0].linear = torch.nn.Sequential()
            model.eval()
        
    elif training_type == 'clean':

        if in_dataset == 'cifar10':
            model_temp = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True).to(device)

        elif in_dataset == 'cifar100':
            model_temp = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True).to(device)
        
        model_temp.classifier = torch.nn.Sequential()
        model_temp.eval()
        
        model = normalizer(model_temp)
    
    return model