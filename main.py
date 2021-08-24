import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.datasets as dset

from model import Generator, Discriminator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from panel_data import PanelDataLoader

manualSeed = 999
print("Random Seed: {}".format(manualSeed))
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "../data/panel_train_ver1/"
workers = 4
batch_size = 8
image_size = 183
nc = 3  # Number of channels
nz = 100  # Size of z latent vetector
ngf = 183  # Size of feature maps in generator
ndf = 183  # Size of feature maps in discriminator
lr = 0.0002
beta1 = 0.5  # hyper parameter for Adam optim
ngpu = 1
num_epochs = 100

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
writer = SummaryWriter(log_dir="runs/experiment2")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def train(train_dataloader):
    netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
    netD = Discriminator(ngpu=ngpu, nc=nc, ndf=ndf).to(device)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(183, nz, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    iters = 0

    print("Start Training")

    for epoch in range(num_epochs):
        tq0 = tqdm(train_dataloader, total=len(train_dataloader))

        for i, itr in enumerate(tq0):
            data, _ = itr

            netD.zero_grad()
            real_cpu = data.to(device)

            b_size = real_cpu.size(0)
            # label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)

            b_size = int(len(output))

            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # noise = torch.randn(b_size, nz, 1, 1, device=device)
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)

            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake

            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake).view(-1)

            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()
            optimizerG.step()

            writer.add_scalars('GAN loss', {'D_loss': errD.item(), 'G_loss': errG.item()}, epoch)

            errors = {
                'Epoch': epoch,
                'Loss_D': errD.item(),
                'Loss_G': errG.item(),
                'D(x)': D_x,
                'D(G(z1))': D_G_z1,
                'D(G(z2))': D_G_z2
            }

            tq0.set_postfix(errors)

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()

                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    torch.save({
        'gen_state_dict': netG.state_dict(),
        'dis_state_dict': netD.state_dict(),
        'gen_optimizer_state_dict': optimizerG.state_dict(),
        'dis_optimizer_state_dict': optimizerD.state_dict(),
    }, './checkpoint/best_weight.pth')


def make_file_list(image_path):
    train_img_list = list()

    for img in os.listdir(image_path):
        train_img_list.append(image_path + img)

    return train_img_list


if __name__ == "__main__":
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # dataset = PanelDataLoader(file_list=make_file_list(dataroot), transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    print("Num of train data: {}".format(len(dataset)))

    train(dataloader)
    print("Training finish!")

# # Showing train dataset
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8, 8))q
# plt.axis("off")
# plt.title("Training Images")
# plt_img = np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0))
# plt.imshow(plt_img)
# plt.show()
