import random
import numpy as np
from einops import rearrange
import torch
from torch import nn
from utils import init_weights

# Fully Rotation Invariant Convolutional Attention MLP
class MLP(nn.Module):
    def __init__(self, num_channels, hidden_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(nn.Linear(num_channels, hidden_size),
                                nn.GELU(),
                                nn.Linear(hidden_size, num_channels))
        init_weights(self.modules())

    def forward(self, x):
        return self.fc(x)

class Attention(nn.Module):
    def __init__(self, heads, hidden_size):
        super(Attention, self).__init__()
        self.heads = heads
        self.dim = hidden_size*4
        self.dim_head = int(self.dim/self.heads)
        self.scale = self.dim_head**-0.5
        self.to_qkv = nn.Linear(hidden_size, hidden_size*3)
        self.to_out = nn.Linear(hidden_size, hidden_size)
        init_weights(self.modules())

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3,-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d',
                                          h=self.heads), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j',q,k)
        dots *= self.scale
        attention = dots.softmax(-1)
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out += x
        return out

class Convolution(nn.Module):
    def __init__(self, num_channels, expansion=4):
        super(Convolution, self).__init__()
        self.downsample1 = nn.Sequential(nn.Conv2d(num_channels,
                                                   num_channels*expansion,
                                                   kernel_size=3, bias=False),
                                         nn.BatchNorm2d(
                                             num_channels*expansion),
                                         nn.LeakyReLU(0.2),
                                         nn.MaxPool2d(kernel_size=3, stride=2))
        self.downsample2 = nn.Sequential(nn.Conv2d(num_channels*expansion,
                                                   num_channels*expansion**2,
                                                   kernel_size=3, bias=False),
                                         nn.BatchNorm2d(
                                             num_channels*expansion**2),
                                         nn.LeakyReLU(0.2),
                                         nn.MaxPool2d(kernel_size=1, stride=2))
        self.downsample3 = nn.Sequential(nn.Conv2d(num_channels*expansion**2,
                                                   num_channels*expansion,
                                                   kernel_size=1, bias=False),
                                         nn.BatchNorm2d(
                                             num_channels*expansion),
                                         nn.LeakyReLU(0.2),
                                         nn.MaxPool2d(kernel_size=1, stride=2))
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(num_channels*expansion,
                               num_channels*expansion**2,
                               kernel_size=2,stride=2,bias=False),
            nn.BatchNorm2d(num_channels*expansion**2),
            nn.LeakyReLU(0.2))
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(num_channels*expansion**2,
                               num_channels*expansion,
                               kernel_size=4,stride=2,bias=False),
            nn.BatchNorm2d(num_channels*expansion),
            nn.LeakyReLU(0.2))
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(num_channels*expansion,
                               num_channels,
                               kernel_size=6,stride=2,bias=False),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2))
        init_weights(self.modules())

    def forward(self, x):
        x1 = self.downsample1(x)
        x2 = self.downsample2(x1)
        x3 = self.downsample3(x2)
        x4 = self.upsample1(x3)
        x5 = self.upsample2(x4)
        out = self.upsample3(x5)
        return out

class ConvolutinalAttentionMLP(nn.Module):
    def __init__(self, num_channels, image_size, patch_size, num_classes,
                 hidden_size, blocks = 4, use_convolution=False,
                 use_attention=False):
        super(ConvolutinalAttentionMLP, self).__init__()
        self.use_attention = use_attention
        self.use_convolution = use_convolution
        self.heads = 4
        self.patch_size = patch_size
        base_patch_size = image_size // 2
        scale = base_patch_size//patch_size
        self.embed = nn.Linear(num_channels*patch_size**2*scale**2,
                               hidden_size)
        self.mlp_block = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_size),
                          MLP(hidden_size, hidden_size*2)) \
            for _ in range(blocks)])
        if use_convolution:
            self.convolution_block = nn.ModuleList([
                Convolution(num_channels) for _ in range(blocks)])
        if use_attention:
            self.attention_block = nn.ModuleList([
                Attention(self.heads, hidden_size) for _ in range(blocks)])
        self.fc1 = nn.Linear(hidden_size*4, hidden_size)
        self.relu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        if self.use_convolution:
            for convolution in self.convolution_block:
                x = x + convolution(x)
        b, c, h, w = x.shape
        base_patch_size = h//2
        self.patch_dim = h//self.patch_size
        if self.patch_dim > 2:
            self.scale = base_patch_size//self.patch_size
        x = x.view(b, c, h//self.patch_size, self.patch_size, w//self.patch_size,
                    self.patch_size)
        x = x.permute(0,2,4,1,3,5).reshape(b,-1,c*self.patch_size**2)
        x = self.mask_and_rotation(x)
        if self.patch_dim > 2:
            x = x.reshape(b,-1,c*self.patch_size**2*self.scale**2)
        x = self.embed(x)
        for mlp in self.mlp_block:
            x = x + mlp(x)
        if self.use_attention:
            for attention in self.attention_block:
                x = x + attention(x)
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    def mask_and_rotation(self, images):
        device = images.get_device()
        images = images.cpu().detach().numpy()
        for i in range(0, images.shape[0]):
            for j in range(0, images.shape[1]):
                x = j % self.patch_dim
                y = j // self.patch_dim
                rnd = random.uniform(0,10)
                if rnd > 6:
                    images[i][y*self.patch_dim:(y+1)*self.patch_dim,
                              x*self.patch_dim:(x+1)*self.patch_dim] *= 0
                if rnd < 4:
                    rotation = random.choice([90,180,270])
                    if rotation == 90: r = 3
                    elif rotation == 180: r = 2
                    elif rotation == 270: r = 1
                    np.rot90(images[i][y*self.patch_dim:(y+1)*self.patch_dim,
                                       x*self.patch_dim:(x+1)*self.patch_dim],r)
        images = torch.from_numpy(images).to(device)
        return images
