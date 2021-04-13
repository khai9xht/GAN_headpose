from tqdm import tqdm

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


def train_pose(hp_net, imgs, labels, criterion, optim):
    optim.zero_grad()
    pred = hp_net(imgs)
    loss = criterion(pred, labels)
    optim.step()
    return loss


def train_st1(gen, dis, hp_net, dataloaders,optims, criterions, schedulers, epochs):
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
            loss = train_pose(hp_net, real_imgs, real_labels, hp_loss, h_optim)
            HP_total_loss += loss

        HP_total_loss.backward()
        h_sched.step()

def train_st2(gen, dis, hp_net, dataloaders,optims, criterions, schedulers, epochs):
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

        total_loss = 0
        for iteration, batch in enumerate(Rdataloader):
            real_imgs, real_labels = batch
            loss = train_d(dis, gan_loss, d_optim, real_imgs, real_labels)
            total_loss += loss

        for iteration, batch in enumerate(Fdataloader):
            fake_vectors, fake_labels = batch
            fake_imgs = gen(fake_vectors)
            loss = train_d(dis, gan_loss, d_optim, fake_imgs, fake_labels)
            total_loss += loss

        total_loss.forward()
        d_sched.step()


        # Train generator model
        for param in gen.parameters():
            param.requires_grad = True
        for param in dis.parameters():
            param.requires_grad = False
        for param in hp_net.parameters():
            param.requires_grad = False

        total_loss = 0
        for iteration, batch in enumerate(Fdataloader):
            fake_vectors, fake_labels = batch
            fake_imgs = gen(fake_vectors)
            loss = train_d(dis, gan_loss, d_optim, fake_imgs, fake_labels)
            total_loss += loss
            fake_imgs = fake_imgs.view(gen.img_shape)
            pred_pose = hp_net(fake_imgs)
            total_loss += hp_loss(pred_pose, fake_vectors)

        total_loss.backward()


        # Train Head pose model
        for param in hp_net.parameters():
            param.requires_grad = True

        HP_total_loss = 0
        for iteration, batch in enumerate(HPdataloader):
            real_imgs, real_labels = batch
            loss = train_pose(hp_net, real_imgs, real_labels, hp_loss, h_optim)
            HP_total_loss += loss

        HP_total_loss.backward()
        h_sched.step()

