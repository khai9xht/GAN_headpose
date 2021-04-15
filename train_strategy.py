from tqdm import tqdm
import torch
from utils import tensor_convertListAngleToVector

def train_d(dis, criterion, d_optim, imgs, labels):
    d_optim.zero_grad()
    pred = dis(imgs.float())
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


def train_pose(hp_net, imgs, labels, criterion, optim):
    optim.zero_grad()
    pred = hp_net(imgs)
    loss = criterion(pred, labels)
    optim.step()
    return loss


def train_st1(gen, dis, hp_net, dataloaders, optims, criterions, schedulers, epochs):
    g_optim, d_optim, h_optim = optims
    gan_loss, hp_loss = criterions
    g_sched, d_sched, h_sched = schedulers
    Fdataloader, Rdataloader, HPdataloader = dataloaders

    epoch = 0
    with tqdm(total=epochs, desc=f"Epoch {epoch + 1}/{epochs}", postfix=dict, mininterval=0.3) as pbar:

        # Train discriminator model
        for param in gen.parameters():
            param.requires_grad = False
        for param in dis.parameters():
            param.requires_grad = True
        
        Gan_total_loss = []
        for iteration, batch in enumerate(Rdataloader):
            real_imgs, real_labels = batch[0], batch[1]
            real_imgs = real_imgs.view((real_imgs.size()[0], -1))
            d_loss = train_d(dis, gan_loss, d_optim, real_imgs, real_labels)
            Gan_total_loss.append(d_loss)

        for iteration, batch in enumerate(Fdataloader):
            fake_vectors, fake_labels = batch[0], batch[1]
            fake_imgs = gen(fake_vectors)
            g_loss = train_d(dis, gan_loss, d_optim, fake_imgs, fake_labels)
            Gan_total_loss.append(g_loss)

        # Gan_total_loss = sum(Gan_total_loss)
        # Gan_total_loss.backward()


        # Train generator model
        for param in gen.parameters():
            param.requires_grad = True
        for param in dis.parameters():
            param.requires_grad = False

        for iteration, batch in enumerate(Fdataloader):
            fake_vectors, fake_labels = batch[0], batch[1]
            fake_imgs = gen(fake_vectors)
            g_loss = train_d(dis, gan_loss, d_optim, fake_imgs, fake_labels)
            Gan_total_loss.append(g_loss)

        Gan_total_loss = sum(Gan_total_loss)
        Gan_total_loss.backward()


        # Train head pose model
        HP_total_loss = []
        for iteration, batch in enumerate(HPdataloader):
            real_imgs, real_labels = batch[0], batch[1]
            h_loss = train_pose(hp_net, real_imgs, real_labels, hp_loss, h_optim)
            HP_total_loss.append(h_loss)

        HP_total_loss = sum(HP_total_loss)
        HP_total_loss.backward()

        g_sched.step()
        d_sched.step()
        h_sched.step()

def train_st2(gen, dis, hp_net, dataloaders,optims, criterions, schedulers, epochs):
    g_optim, d_optim, h_optim = optims
    gan_loss, hp_loss = criterions
    g_sched, d_sched, h_sched = schedulers
    Fdataloader, Rdataloader, HPdataloader = dataloaders

    epoch = 0
    with tqdm(total=epochs, desc=f"Epoch {epoch + 1}/{epochs}", postfix=dict, mininterval=0.3) as pbar:

        # Train discriminator model
        for param in gen.parameters():
            param.requires_grad = False
        for param in dis.parameters():
            param.requires_grad = True
        for param in hp_net.parameters():
            param.requires_grad = False

        total_loss = []
        for iteration, batch in enumerate(Rdataloader):
            real_imgs, real_labels = batch[0], batch[1]
            real_imgs = real_imgs.view((real_imgs.size()[0], -1))
            d_loss = train_d(dis, gan_loss, d_optim, real_imgs, real_labels)
            total_loss.append(d_loss)

        for iteration, batch in enumerate(Fdataloader):
            fake_vectors, fake_labels = batch[0], batch[1]
            fake_imgs = gen(fake_vectors)
            g_loss = train_d(dis, gan_loss, d_optim, fake_imgs, fake_labels)
            total_loss.append(g_loss)


        # Train generator model
        for param in gen.parameters():
            param.requires_grad = True
        for param in dis.parameters():
            param.requires_grad = False
        for param in hp_net.parameters():
            param.requires_grad = False


        for iteration, batch in enumerate(Fdataloader):
            fake_vectors, fake_labels = batch[0], batch[1]
            fake_imgs = gen(fake_vectors)
            g_loss = train_d(dis, gan_loss, d_optim, fake_imgs, fake_labels)
            total_loss.append(g_loss)
            fake_imgs = fake_imgs.view(64, 3, 64, 64).contiguous()
            pred_pose = hp_net(fake_imgs)
            rotateMatrixs = tensor_convertListAngleToVector(pred_pose[:, 0], pred_pose[:, 1], pred_pose[:, 2])
            total_loss.append(hp_loss(rotateMatrixs, fake_vectors))


        # Train Head pose model
        for param in gen.parameters():
            param.requires_grad = False
        for param in dis.parameters():
            param.requires_grad = False
        for param in hp_net.parameters():
            param.requires_grad = True

        for iteration, batch in enumerate(HPdataloader):
            real_imgs, real_labels = batch[0], batch[1]
            h_loss = train_pose(hp_net, real_imgs, real_labels, hp_loss, h_optim)
            total_loss.append(h_loss)

        total_loss = sum(total_loss)
        total_loss.backward()

        g_sched.step()
        d_sched.step()
        h_sched.step()

