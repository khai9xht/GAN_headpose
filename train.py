import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm

from model.Gan import Generator, Discriminator
from model.fsanet import FSANet


def train_d(dis, criterion, d_optim, imgs, labels):
    d_optim.zero_grad()

    pred = dis(imgs)
    loss = criterion(pred, labels)

    d_optim.step()
    return loss


def train_g(dis, gen, g_optim, criterion, FalseData, falseLabel):
    g_optim.zero_grad()

    gen_img = gen(FalseData)
    falsePred = dis(gen_img)
    loss = criterion(falsePred, falseLabel)
    g_optim.step()
    return loss


def train_pose(pose_net, imgs, labels, criterion, optim):
    optim.zero_grad()

    pred = pose_net(imgs)
    loss = criterion(pred, labels)
    optim.step()
    return loss


def train(gen, dis, pose_net, dataloaders,optims, criterions, schedulers, epochs):
    g_optim, d_optim, h_optim = optims
    gan_loss, hp_loss = criterions
    g_ched, d_sched, h_sched = schedulers
    Fdataloader, Rdataloader, HPdataloader = dataloaders

    for epoch in tqdm(range(epochs)):

        # Train discriminator model
        for param in gen.parameters():
            param.requires_grad = False
        for param in dis.parameters():
            param.requires_grad = True
        GAN_total_loss = 0
        for iteration, batch in enumerate(Rdataloader):
            real_imgs, real_labels = batch
            loss = train_d(dis, gan_loss, d_optim, real_imgs, real_labels)
            GAN_total_loss += loss

        for iteration, batch in enumerate(Fdataloader):
            fake_vectors, fake_labels = batch
            fake_imgs = gen(fake_vectors)
            loss = train_d(dis, gan_loss, d_optim, fake_imgs, fake_labels)
            GAN_total_loss += loss
        GAN_total_loss.forward()
        d_sched.step()

        # Train generator model
        for param in gen.parameters():
            param.requires_grad = True
        for param in dis.parameters():
            param.requires_grad = False
        GAN_total_loss = 0
        for iteration, batch in enumerate(Fdataloader):
            fake_vectors, fake_labels = batch
            fake_imgs = gen(fake_vectors)
            loss = train_d(dis, gan_loss, d_optim, fake_imgs, fake_labels)
            GAN_total_loss += loss
        GAN_total_loss.backward()

        # Train head pose model
        HP_total_loss = 0
        for iteration, batch in enumerate(HPdataloader):
            real_imgs, real_labels = batch
            loss = train_pose(pose_net, real_imgs, real_labels, hp_loss, h_optim)
            HP_total_loss += loss
        HP_total_loss.backward()
        h_sched.step()



if __name__ == "__main__":
    latent_dim =110
    img_shape = (64, 64, 3)
    gen = Generator(latent_dim, img_shape)
    dis = Discriminator(img_shape)
    criterion = nn.BCELoss()
    lr = 1e-4
    g_optim = torch.optim.Adam(gen.parameters(), lr=lr)
    d_optim = torch.optim.Adam(dis.parameters(), lr=lr)
    g_sched = torch.optim.lr_scheduler.StepLR(g_optim, step_size=1, gamma=0.97)
    d_sched = torch.optim.lr_scheduler.StepLR(d_optim, step_size=1, gamma=0.97)

    batch_size = 32
    epochs = 100
