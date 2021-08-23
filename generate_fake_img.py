import torch
import albumentations.pytorch
import albumentations.augmentations
import matplotlib.pyplot as plt

from model import Generator

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean

image_size = 64
nc = 3  # Number of channels
nz = 100  # Size of z latent vetector
ngf = 64  # Size of feature maps in generator
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

transform = albumentations.Compose([
    albumentations.Resize(height=image_size, width=image_size),
    albumentations.CenterCrop(height=image_size, width=image_size),
    albumentations.pytorch.ToTensorV2(),
    albumentations.augmentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


class GenerateFace(object):
    def __init__(self, weight_path):
        self.weight = torch.load(weight_path)
        self.G_model = Generator(ngpu, nz, ngf, nc)
        self.G_model.load_state_dict(self.weight['gen_state_dict'])
        self.G_model.eval().to(device=device)

    def generate(self):
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake = self.G_model(noise)
        return fake

if __name__ == "__main__":
    Generate_obj = GenerateFace('./checkpoint/best_weight.pth')
    output = Generate_obj.generate()

    for i in range(100):
        output = Generate_obj.generate()
        fake_image = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()
        plt.imsave("./fake_image/"+str(i)+".png", fake_image)

