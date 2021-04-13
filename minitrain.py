from config import CONFIG
from train_strategy import train_st1, train_st2
from model.Gan import Generator, Discriminator
from model.fsanet import FSANet

import torch.optim as optim
import torch.nn as nn
from torch,utils,data import DataLoader

gen = Generator(CONFIG["vector dim"], CONFIG["image shape"])
dis = Discriminator(CONFIG["image shape"])
hp_net = FSANet()

gan_criterion = nn.MSELoss()
hp_criterion = nn.MSELoss()
criterions = [gan_criterion, hp_criterion]

g_optim = optim.SGD(gen.parameters(), lr=CONFIG["learning rate"])
d_optim = optim.SGD(dis.parameters(), lr=CONFIG["learning rate"])
h_optim = optim.SGD(hp_net.parameters(), lr=CONFIG["learning rate"])
optims = [g_optim, d_optim, h_optim]

g_sched = optim.lr_scheduler.StepLR(g_optim, step_size=1, gamma=0.97)
d_sched = optim.lr_scheduler.StepLR(d_optim, step_size=1, gamma=0.97)
h_sched = optim.lr_scheduler.StepLR(h_optim, step_size=1, gamma=0.97)
scheds = [g_sched, d_sched, h_sched]

fake_dataset = None
face_dataset = None
hp_dataset = None

fake_dataloader = DataLoader(fake_dataset)

train_st1(gen, dis, hp_net, )


