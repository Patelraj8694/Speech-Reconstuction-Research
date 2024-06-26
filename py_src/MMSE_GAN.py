import numpy as np
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets, models
from torch import Tensor

import visdom
import math
import matplotlib.pyplot as plt 
import scipy
from scipy import io as sio
from scipy.io import savemat
from scipy.io import loadmat

from dataloaders import parallel_dataloader, non_parallel_dataloader
from networks import dnn_generator, dnn_discriminator
from utils import *

import argparse

# Training Function
def training(data_loader, n_epochs):
    Gnet.train()
    Dnet.train()
    
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0).type(torch.FloatTensor)).to(device)
        b = Variable(b.squeeze(0).type(torch.FloatTensor)).to(device)

        valid = Variable(Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
        
        # Update G network
        optimizer_G.zero_grad()
        Gout = Gnet(a)
        
        G_loss = adversarial_loss(Dnet(Gout), valid) + mmse_loss(Gout, b)
        
        G_loss.backward()
        optimizer_G.step()


        # Update D network
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(Dnet(b), valid)
        fake_loss = adversarial_loss(Dnet(Gout.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2
        
        D_loss.backward()
        optimizer_D.step()
        
        
        print ("[Epoch: %d] [Iter: %d/%d] [D loss: %f] [G loss: %f]" % (n_epochs, en, len(data_loader), D_loss, G_loss.cpu().data.numpy()))
 
# Validation function that also calculates MCD
def validating(data_loader):
    Gnet.eval()
    Dnet.eval()
    Grunning_loss = 0
    Drunning_loss = 0
    mcd_values = []
    
    for en, (a, b) in enumerate(data_loader):
        a = Variable(a.squeeze(0).type(torch.FloatTensor)).to(device)
        b = Variable(b.squeeze(0).type(torch.FloatTensor)).to(device)

        valid = Variable(torch.Tensor(a.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(a.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
        
        Gout = Gnet(a)
        G_loss = adversarial_loss(Dnet(Gout), valid) + mmse_loss(Gout, b)
        Grunning_loss += G_loss.item()

        real_loss = adversarial_loss(Dnet(b), valid)
        fake_loss = adversarial_loss(Dnet(Gout.detach()), fake)
        D_loss = (real_loss + fake_loss) / 2
        Drunning_loss += D_loss.item()
        
        # Calculate MCD
        b_np = b.cpu().data.numpy()
        Gout_np = Gout.cpu().data.numpy()
        mcd = np.mean([logSpecDbDist(Gout_np[i][1:], b_np[i][1:]) for i in range(len(b_np))])
        mcd_values.append(mcd)
    
    avg_mcd = np.mean(mcd_values)
    # Update Visdom for MCD
    # viz.line(Y=np.array([avg_mcd]), X=np.array([epoch]), win='mcd_plot', update='append' if epoch > 0 else 'replace', opts=dict(title='MCD by Epoch'))
    return Drunning_loss/(en+1), Grunning_loss/(en+1), avg_mcd

# Training function updated to include MCD and maintain existing functionality
def do_training():
    epoch = args.epoch

    dl_arr = []
    gl_arr = []
    mcd_arr = []
    for ep in range(epoch):

        training(train_dataloader, ep+1)
        if (ep+1)%args.checkpoint_interval==0:
            torch.save(Gnet, join(checkpoint,"gen_Ep_{}.pth".format(ep+1)))
            torch.save(Dnet, join(checkpoint,"dis_Ep_{}.pth".format(ep+1)))


        if (ep+1)%args.validation_interval==0:
            dl,gl,mcd = validating(val_dataloader)
            
            print("D_loss: " + str(dl) + " G_loss: " + str(gl))
            
            dl_arr.append(dl)
            gl_arr.append(gl)
            mcd_arr.append(mcd)
            
            if ep == 0:
                gplot = viz.line(Y=np.array([gl]), X=np.array([ep]), opts=dict(title='Generator'))
                dplot = viz.line(Y=np.array([dl]), X=np.array([ep]), opts=dict(title='Discriminator'))
                mplot = viz.line(Y=np.array([mcd]), X=np.array([ep]), opts=dict(title='MCD'))
            else:
                viz.line(Y=np.array([gl]), X=np.array([ep]), win=gplot, update='append')
                viz.line(Y=np.array([dl]), X=np.array([ep]), win=dplot, update='append')
                viz.line(Y=np.array([mcd]), X=np.array([ep]), win=mplot, update='append')
    
    savemat(checkpoint+"/"+str('discriminator_loss.mat'),  mdict={'foo': dl_arr})
    savemat(checkpoint+"/"+str('generator_loss.mat'),  mdict={'foo': gl_arr})
    savemat(checkpoint+"/"+str('mcd.mat'),  mdict={'foo': mcd_arr})

    plt.figure(1)
    plt.plot(dl_arr)
    plt.savefig(checkpoint+'/discriminator_loss.png')
    plt.figure(2)
    plt.plot(gl_arr)
    plt.savefig(checkpoint+'/generator_loss.png')
    plt.figure(3)
    plt.plot(mcd_arr)
    plt.savefig(checkpoint+'/mcd.png')

def do_testing():
    print("Testing")
    save_folder = args.save_folder
    test_folder_path=args.test_folder

    dirs = listdir(test_folder_path)
    Gnet = torch.load(join(checkpoint,"gen_Ep_{}.pth".format(args.test_epoch))).to(device)

    for i in dirs:
        
        # Load the .mcc file
        d = read_mcc(join(test_folder_path, i))

        a = torch.from_numpy(d)
        a = Variable(a.squeeze(0).type('torch.FloatTensor')).to(device)
        
        Gout = Gnet(a)

        savemat(join(save_folder,'{}.mat'.format(i[:-4])),  mdict={'foo': Gout.cpu().data.numpy()})

def give_MCD():
    Gnet = torch.load(join(checkpoint,"gen_Ep_{}.pth".format(args.test_epoch))).to(device)
    mcd = []

    for en, (a, b) in enumerate(val_dataloader):
        a = Variable(a.squeeze(0).type(torch.FloatTensor)).to(device)
        b = b.cpu().data.numpy()[0]

        Gout = Gnet(a).cpu().data.numpy()

        ans = 0
        for k in range(Gout.shape[0]):
            ans = logSpecDbDist(Gout[k][1:],b[k][1:])
            mcd.append(ans)

    mcd = np.array(mcd)
    print(np.mean(mcd))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Training methodology for Whisper-to-Normal Speech Conversion")
    parser.add_argument("-np", "--nonparallel", type=bool, default=False, help="Parallel training or non-parallel?")
    parser.add_argument("-dc", "--dnn_cnn", type=str, default='dnn', help="DNN or CNN architecture for generator and discriminator?")
    parser.add_argument("-tr", "--train", action="store_true", help="Want to train?")
    parser.add_argument("-te", "--test", action="store_true", help="Want to test?")
    parser.add_argument("-m", "--mcd", action="store_true", help="Want MCD value?")
    parser.add_argument("-ci", "--checkpoint_interval", type=int, default=5, help="Checkpoint interval")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="Number of Epochs")
    parser.add_argument("-et", "--test_epoch", type=int, default=100, help="Epochs to test")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-vi", "--validation_interval", type=int, default=1, help="Validation Interval")
    parser.add_argument("-mf", "--mainfolder", type=str, default="../dataset/features/US_102/batches/mcc/", help="Main folder path to load MCC batches")
    parser.add_argument("-vf", "--validation_folder", type=str, default="../dataset/features/US_102/batches/mcc/", help="Validation folder path to load MCC batches")
    parser.add_argument("-cf", "--checkpoint_folder", type=str, default="../results/checkpoints/mcc/", help="Checkpoint saving path for MCC features")
    parser.add_argument("-sf", "--save_folder", type=str, default="../results/mask/mcc/", help="Saving folder for converted MCC features")
    parser.add_argument("-tf", "--test_folder", type=str, default="../dataset/features/US_102/Whisper/mcc/", help="Input whisper mcc features for testing")

    args = parser.parse_args()


    
    # Connect with Visdom for the loss visualization
    viz = visdom.Visdom()

    # Path where you want to store your results        
    mainfolder = args.mainfolder
    checkpoint = args.checkpoint_folder
    validation = args.validation_folder

    # Training Data path
    if args.nonparallel:
        custom_dataloader = non_parallel_dataloader
    else:
        custom_dataloader = parallel_dataloader

    traindata = custom_dataloader(folder_path=mainfolder)
    train_dataloader = DataLoader(dataset=traindata, batch_size=1, shuffle=True, num_workers=0)  # For windows keep num_workers = 0


    # Path for validation data
    valdata = custom_dataloader(folder_path=validation)
    val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=True, num_workers=0)  # For windows keep num_workers = 0


    # Loss Functions
    adversarial_loss = nn.BCELoss()
    mmse_loss = nn.MSELoss()

    ip_g = 40 # MCEP feature dimentions
    op_g = 40 # MCEP feature dimentions
    ip_d = 40 # MCEP feature dimentions
    op_d = 1


    # Check for Cuda availability
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # Initialization
    if args.dnn_cnn == "dnn":
        Gnet = dnn_generator(ip_g, op_g, 512, 512, 512).to(device)
        Dnet = dnn_discriminator(ip_d, op_d, 512, 512, 512).to(device)


    # Initialize the optimizers
    optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=args.learning_rate)
    optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=args.learning_rate)

    if args.train:
        do_training()
    if args.test:
        do_testing()
    if args.mcd:
        give_MCD()
