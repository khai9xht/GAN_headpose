import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader 
import torch.nn.functional as F

from tqdm import tqdm

from GAN import Generator, Discriminator


def train_d(dis, opt, criterion, sched, TrueData, FalseData, trueLabel, falseLabel):
    trueSize = TrueData.size(0)
    falseSize = FalseData.size(0)

    opt.zero_grad()

    truePred = dis(TrueData)
    trueError = criterion(truePred, trueLabel)
    trueError.backward()

    falsePred = dis(FalseData)
    falseError = criterion(falsePred, falseLabel)
    falseError.backward()

    opt.step()
    sched.step()

def train_g(gen, opt, sched, FalseData, falseLabel):
    opt.zero_grad()

    pred = gen(FalseData)


def train(gen, dis, dataloader, epochs, criterion, g_optim, d_optim, g_sched, d_sched):
    total_loss = 0
    val_loss = 0
    for epoch in tqdm(range(epochs)):
        for iteration, batch in enumerate(dataloader):
            


if __name__ == "__main__":
    latent_dim =100
    img_shape = (64, 64, 3)
    gen = Generator(latent_dim, img_shape)
    dis = Discriminator(img_shape)
    criterion = nn.BCELoss()
    lr = 2e-4
    g_optim = torch.optim.Adam(gen.parameters(), lr=lr)
    d_optim = torch.optim.Adam(dis.parameters(), lr=lr)
    g_sched = torch.optim.lr_scheduler.StepLR(g_sched, step_size=1, gamma=0.97)
    d_sched = torch.optim.lr_scheduler.StepLR(d_sched, step_size=1, gamma=0.97)

    batch_size = 32
    epochs = 100