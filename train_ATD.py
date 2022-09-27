import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
import os, copy

from utils import fix_random_seed, get_feature_extractor_model
from data.closed_set import get_in_training_loaders
from data.open_set import get_out_training_loaders
from pgd_attack import attack_pgd
from models.DCGAN import Generator_fea, Discriminator_fea, wrapper_fea, Generator_pix, Discriminator_pix, weights_init


os.environ['TORCH_HOME'] = 'models/'


#get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='fea', type=str, choices={'fea', 'pix'})
    parser.add_argument('--training_type', default='adv', type=str, choices={'clean', 'adv'})
    parser.add_argument('--in_dataset', default='cifar10', type=str, choices={'cifar10', 'cifar100', 'TI'})
    
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--eps', default=8/255, type=float)
    parser.add_argument('--attack_iters', default=10, type=int)
    
    parser.add_argument('--run_name', default='test', type=str)
    parser.add_argument('--seed', default=0, type=int)
    
  
    return parser.parse_args()

args = get_args()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_type = args.model_type
training_type = args.training_type
in_dataset = args.in_dataset

alpha = args.alpha
batch_size = args.batch_size
num_epochs = args.num_epochs
eps= args.eps
attack_iters = args.attack_iters
alp = 2.5*eps/attack_iters
clean_val = alpha<=0.5

run_name = args.run_name
print('Run name:', run_name)

#set random seed
seed = args.seed
fix_random_seed(seed)


#define deture extractor model
model = get_feature_extractor_model(training_type, in_dataset)

#in dataset
trainloader, valloader = get_in_training_loaders(in_dataset, batch_size)

#out dataset
trainloader_out, valloader_out = get_out_training_loaders(batch_size)


#Model DCGAN
# Number of channels in the training images. For color images this is 3
if model_type == 'fea':
    nc = 512
elif model_type == 'pix':
    nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available.
ngpu = 1


if model_type == 'fea':
    netG = Generator_fea(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
    netD = Discriminator_fea(ngpu=ngpu, nc=nc, ndf=ndf).to(device)
    
    forward_pass = wrapper_fea(model, netD)

elif model_type == 'pix':
    netG = Generator_pix(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
    netD = Discriminator_pix(ngpu=ngpu, nc=nc, ndf=ndf).to(device)
    
    forward_pass = netD


#weights_init
netD.apply(weights_init)
netG.apply(weights_init)


#############################

#train
# Initialize BCELoss function
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

lr = 1e-4

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr/1.5, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

save_dir = 'checkpoints/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

#Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
best_auc = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
       
    for i, ((x, y), (x_out, y_out)) in enumerate(zip(trainloader, trainloader_out), 0):

        # Format batch
        x = x.to(device)
        x_out = x_out.to(device)
        b_size = x.size(0)
        
        # Attack
        netD.eval()
        delta_x = torch.zeros_like(x)
        if training_type == 'adv':  
            delta_x = attack_pgd(forward_pass, x, torch.ones_like(y, dtype=torch.float32).to(device), epsilon=eps, alpha=alp, attack_iters=attack_iters)
 
        delta_x_out = torch.zeros_like(x_out)
        if training_type == 'adv':  
            delta_x_out = attack_pgd(forward_pass, x_out, torch.zeros_like(y_out, dtype=torch.float32).to(device), epsilon=eps, alpha=alp, attack_iters=attack_iters)

        #Update D
        netG.eval()
        netD.train()
        netD.zero_grad()
        
        label_batch_real = torch.full((b_size,), real_label, device=device).to(torch.float32)
        
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        
        if model_type == 'fea':
            feature_real = model(x+delta_x).unsqueeze(2).unsqueeze(3)
            feture_out = model(x_out+delta_x_out).unsqueeze(2).unsqueeze(3)
        elif model_type == 'pix':
            feature_real = x+delta_x
            feture_out = x_out+delta_x_out
        
        cat_real_fake_out = torch.cat([feature_real, fake, feture_out], dim=0)
        output_real_fake_out = netD(cat_real_fake_out.detach()).view(-1)
        output_real, output_fake, output_out = output_real_fake_out[:b_size], output_real_fake_out[b_size:2*b_size], output_real_fake_out[2*b_size:]
        
        D_x = output_real.mean().item()
        errD_real = criterion(output_real, label_batch_real)      
        
        D_G_z1 = output_fake.mean().item()
        label_batch_fake = torch.full((b_size,), fake_label, device=device).to(torch.float32)
        errD_fake = criterion(output_fake, label_batch_fake)      
        errD_fake = (1-alpha)*errD_fake
        

        label_out = torch.full((x_out.size(0),), fake_label, device=device).to(torch.float32)
        errD_real_out = criterion(output_out, label_out)
            
        errD_real_out = alpha*errD_real_out
        D_x_out = output_out.mean().item()
        errD = errD_real + errD_fake + errD_real_out

        
        errD.backward()
        optimizerD.step()

        #Update G
        netG.train()
        netD.eval()
        netG.zero_grad()
        
        label_batch_fake_real = torch.full((b_size,), real_label, device=device).to(torch.float32)
        
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        output_fake = netD(fake.detach()).view(-1)
        D_G_z2 = output_fake.mean().item()
        errG = criterion(output_fake, label_batch_fake_real)

        errG.backward()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(x_out): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(trainloader),
                     errD.item(), errG.item(), D_x, D_x_out, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or (epoch == num_epochs-1):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
    
    #validation
    netG.eval()
    netD.eval()
    
    scores_in = []
    scores_out = []
    for (x, y) in valloader:
        x = x.to(device)
        if clean_val:
            delta = torch.zeros_like(x)
        else:
          delta = attack_pgd(forward_pass, x, torch.ones_like(y, dtype=torch.float32).to(device), epsilon=eps, alpha=alp, attack_iters=attack_iters)
        output = forward_pass(x+delta).view(-1)        
        scores_in += output.cpu().detach().tolist()
    
    for (x, y) in valloader_out:
        x = x.to(device)
        if clean_val:
            delta = torch.zeros_like(x)
        else:
          delta = attack_pgd(forward_pass, x, torch.zeros_like(y, dtype=torch.float32).to(device), epsilon=eps, alpha=alp, attack_iters=attack_iters)
        output = forward_pass(x+delta).view(-1)        
        scores_out += output.cpu().detach().tolist()
        
        if len(scores_out)>=len(scores_in):
            break
     
    
    onehots = np.array([1]*len(scores_out) + [0]*len(scores_in))
    scores = np.concatenate([scores_out, scores_in],axis=0)
    new_auc = roc_auc_score(onehots, -scores)
    print(new_auc)
    
    if new_auc>best_auc:  
        best_auc = new_auc
        print('checkpoint!')
        
        cur_model_wts = copy.deepcopy(netG.state_dict())
        path_to_save_paramOnly = os.path.join(save_dir, 'GNet_best_' + run_name)
        torch.save(cur_model_wts, path_to_save_paramOnly)

        cur_model_wts = copy.deepcopy(netD.state_dict())
        path_to_save_paramOnly = os.path.join(save_dir, 'DNet_best_' + run_name)
        torch.save(cur_model_wts, path_to_save_paramOnly)
 
if num_epochs>0:       
    cur_model_wts = copy.deepcopy(netG.state_dict())
    path_to_save_paramOnly = os.path.join(save_dir, 'GNet_last_' + run_name)
    torch.save(cur_model_wts, path_to_save_paramOnly)
    
    cur_model_wts = copy.deepcopy(netD.state_dict())
    path_to_save_paramOnly = os.path.join(save_dir, 'DNet_last_' + run_name)
    torch.save(cur_model_wts, path_to_save_paramOnly)

#############################