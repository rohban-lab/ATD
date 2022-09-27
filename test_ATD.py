import torch
import torch.nn as nn

import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score
import argparse
import os

from utils import fix_random_seed, get_feature_extractor_model
from data.closed_set import get_in_testing_loader
from data.open_set import get_out_testing_datasets
from pgd_attack import attack_pgd
from models.DCGAN import Generator_fea, Discriminator_fea, wrapper_fea, Generator_pix, Discriminator_pix, weights_init

os.environ['TORCH_HOME'] = 'models/'

#get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='fea', type=str, choices={'fea', 'pix'})
    parser.add_argument('--training_type', default='adv', type=str, choices={'clean', 'adv'})
    parser.add_argument('--in_dataset', default='cifar10', type=str, choices={'cifar10', 'cifar100', 'TI'})
    parser.add_argument("--out_datasets", nargs='+', default=['mnist', 'tiny_imagenet', 'places', 'LSUN', 'iSUN', 'birds', 'flowers', 'coil'])
    
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eps', default=8/255, type=float)
    parser.add_argument('--attack_iters', default=100, type=int)
    
    parser.add_argument('--run_name', default='test', type=str)
    parser.add_argument('--seed', default=0, type=int)
    
  
    return parser.parse_args()

args = get_args()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_type = args.model_type
training_type = args.training_type
in_dataset = args.in_dataset
out_names = args.out_datasets

batch_size = args.batch_size
eps= args.eps
epsilons = np.array([0, eps])
attack_iters = args.attack_iters

run_name = args.run_name
test_type = 'best_'
save_dir = 'checkpoints/'
print('Run name:', run_name)

if in_dataset == 'TI' and 'tiny_imagenet' in out_names:
    out_names.remove('tiny_imagenet')

#set random seed
seed = args.seed
fix_random_seed(seed)

#define deture extractor model
model = get_feature_extractor_model(training_type, in_dataset)

#in dataset
testloader = get_in_testing_loader(in_dataset, batch_size)

#out datasets
out_names, out_datasets = get_out_testing_datasets(out_names)

print('Out datasets:', out_names)

#Model DCGAN
# Number of channels in the training images. For color images this is 3
if model_type == 'fea':
    nc = 512
elif model_type == 'pix':
    nc = 3

# Size of feature maps in discriminator
ndf = 64

# Number of GPUs available.
ngpu = 1


if model_type == 'fea':
    netD = Discriminator_fea(ngpu=ngpu, nc=nc, ndf=ndf).to(device)
    
    forward_pass = wrapper_fea(model, netD)

elif model_type == 'pix':
    netD = Discriminator_pix(ngpu=ngpu, nc=nc, ndf=ndf).to(device)
    
    forward_pass = netD

#load model
print('\n', test_type)
netD.load_state_dict(torch.load(os.path.join(save_dir, 'DNet_' + test_type + run_name)))
netD.eval()

scores_in = [[] for i in epsilons]
scores_out = [[[] for j in epsilons] for i in out_datasets]

#scores in
for i, eps in enumerate(epsilons):

    alpha = 2.5*eps/attack_iters

    for (x, y) in tqdm.tqdm(testloader, desc=in_dataset+"_"+str(round(eps,3))):
        x = x.to(device)

        if eps == 0:
            delta = torch.zeros_like(x)
        else:
            delta = attack_pgd(forward_pass, x, torch.ones_like(y, dtype=torch.float32).to(device),  epsilon=eps, 
                               alpha=alpha, attack_iters=attack_iters)

        output = forward_pass(x+delta).view(-1) 
        scores_in[i] += output.cpu().detach().tolist()


#scores out
for i, dataset in enumerate(out_datasets):

    for j, eps in enumerate(epsilons):
        alpha = 2.5*eps/attack_iters

        testloader_out = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        for (x, y) in tqdm.tqdm(testloader_out, desc=args.out_datasets[i]+"_"+str(round(eps,3))):
            x = x.to(device)

            if eps == 0:
                delta = torch.zeros_like(x)
            else: 
                delta = attack_pgd(forward_pass, x, torch.zeros_like(y, dtype=torch.float32).to(device), epsilon=eps,
                                   alpha=alpha, attack_iters=attack_iters)

            output = forward_pass(x+delta).view(-1) 
            scores_out[i][j] += output.cpu().detach().tolist()


#auc
for i, score_out_dataset in enumerate(scores_out):

    print('\ndataset:', out_names[i])

    print('\njust in attacked')
    score_out = score_out_dataset[0]
    for k, score_in in enumerate(scores_in):
        onehots = np.array([1]*len(score_out) + [0]*len(score_in))
        scores = np.concatenate([score_out, score_in],axis=0)
        auroc = roc_auc_score(onehots, -scores)
        print('eps=', epsilons[k], ':', auroc)


    print('\njust out attacked')
    score_in = scores_in[0]
    for k, score_out in enumerate(score_out_dataset):
        onehots = np.array([1]*len(score_out) + [0]*len(score_in))
        scores = np.concatenate([score_out, score_in],axis=0)
        auroc = roc_auc_score(onehots, -scores)
        print('eps=', epsilons[k], ':', auroc)

    print('\nboth attacked')
    for k in range(len(scores_in)):
        score_in = scores_in[k]
        score_out = score_out_dataset[k]

        onehots = np.array([1]*len(score_out) + [0]*len(score_in))
        scores = np.concatenate([score_out, score_in],axis=0)
        auroc = roc_auc_score(onehots, -scores)
        print('eps=', epsilons[k], ':', auroc)