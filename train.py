from config import CONFIG
from train_strategy import train_st1, train_st2
from model.Gan import Generator, Discriminator
from model.fsanet import FSANet

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BIWIDataset, FakeDataset, WLPDataset

print("Initialize models")
gen = Generator(CONFIG["vector dims"]*3, CONFIG["image shape"])
dis = Discriminator(CONFIG["image shape"])
hp_net = FSANet()
print("Initialize models successfully")

print("initialize losses")
gan_criterion = nn.MSELoss()
hp_criterion = nn.MSELoss()
criterions = [gan_criterion, hp_criterion]
print("Initialize loss successfully")

print("initialize optimizers")
g_optim = optim.SGD(gen.parameters(), lr=CONFIG["learning rate"])
d_optim = optim.SGD(dis.parameters(), lr=CONFIG["learning rate"])
h_optim = optim.SGD(hp_net.parameters(), lr=CONFIG["learning rate"])
optims = [g_optim, d_optim, h_optim]
print("initialize optimizer successfully")

print("initialize learning schedulers")
g_sched = optim.lr_scheduler.StepLR(g_optim, step_size=1, gamma=0.97)
d_sched = optim.lr_scheduler.StepLR(d_optim, step_size=1, gamma=0.97)
h_sched = optim.lr_scheduler.StepLR(h_optim, step_size=1, gamma=0.97)
scheds = [g_sched, d_sched, h_sched]
print("initialize learning scheduler successfully")

print("initialize datasets")
face_dataset = WLPDataset(CONFIG["face annotate"], CONFIG["image shape"])
fake_dataset = FakeDataset(face_dataset.__len__(), CONFIG["vector dims"])
hp_dataset = BIWIDataset(CONFIG["head pose annotate"], CONFIG["image shape"])
print("initialize dataset successfully")

print("initialize dataloaders")
fake_dataloader = DataLoader(fake_dataset, batch_size=CONFIG["batch size"], num_workers=CONFIG["num workers"], drop_last=True)
face_dataloader = DataLoader(face_dataset, batch_size=CONFIG["batch size"], num_workers=CONFIG["num workers"], drop_last=True)
hp_dataloader = DataLoader(hp_dataset, batch_size=CONFIG["batch size"], num_workers=CONFIG["num workers"], drop_last=True)
dataloaders = [fake_dataloader, face_dataloader, hp_dataloader]
print("initialize dataloaders successfully")

# print("Start training in stage 1")
# train_st1(gen, dis, hp_net, dataloaders, optims, criterions, scheds, CONFIG["epochs in st1"])

print("Start training in stage 2")
train_st2(gen, dis, hp_net, dataloaders, optims, criterions, scheds, CONFIG["epochs in st2"])

