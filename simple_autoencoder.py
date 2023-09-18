import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(1, 28, 28)
    return x.permute(1, 2, 0)

def to_imgs(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x
g
def show(img):
    plt.imshow(to_img(img))
    plt.show()


batch_size = 128

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    
])

dataset = MNIST('./data', transform=img_transform,download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# explore the images
dataloader_iter=iter(dataloader)
imgs,labels=next(dataloader_iter)

idx=80
print(labels[idx])
show(imgs[idx,:,:,:])


class Autoencoder(nn.Module):
    """
    Audoencoder(latent_dim)
    """
    def __init__(self,latent_dim=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# here, the data is flattened to 1d array with 28*28 elements
# autoencoder.encoder(imgs.view(imgs.size(0), -1)).shape
num_epochs = 100
learning_rate = 1e-3


autoencoder=Autoencoder(2)
autoencoder=autoencoder.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

# imgs.view(imgs.size(0), -1).cuda())
os.mkdir("./mlp_img")

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1).cuda()
        # ===================forward=====================
        output = autoencoder(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))
    pic = to_imgs(output.cpu().data)
    save_image(pic, './mlp_img/output_{}.png'.format(epoch))
    pic = to_imgs(img.cpu().data)
    save_image(pic, './mlp_img/input_{}.png'.format(epoch))


torch.save(autoencoder.state_dict(),"./autoencoder_v1.dat")

iter_=iter(dataloader)
imgs,labels=next(iter_)
imgs = imgs.view(img.size(0), -1).cuda()
# autoencoder(imgs)
latent=autoencoder.encoder(imgs)

import matplotlib.pyplot as plt

labelToColor={0:"red",1:"black",2:"blue",3:"green",4:"red",5:"pink",6:"purple",7:"gray",8:"orange",9:"cyan"}


plt.clf()


for index,point in enumerate(latent):
    label=labels[index].item()
    print(point)
    if(label in labelToColor):
        plt.plot(point.data[0].item(),
                 point.data[1].item(), marker="o", markersize=20, markeredgecolor="red", markerfacecolor=labelToColor[label])

plt.show()



