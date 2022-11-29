import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils.dataloader import get_loader

batch_size = 64
train_size = 224
train_path = './dataset/sekkai_TrainDataset'
val_path = './dataset/sekkai_ValDataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataloaderの作成
image_root = '{}/images/'.format(train_path)
gt_root = '{}/masks/'.format(train_path)
image_root_val = '{}/images/'.format(val_path)
gt_root_val = '{}/masks/'.format(val_path)
train_loader = get_loader(image_root, gt_root, batchsize=batch_size, trainsize=train_size)
val_loader = get_loader(image_root_val, gt_root_val, batchsize=batch_size, trainsize=train_size, phase='val')
# 学習

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


# class VAE(nn.Module):
#     def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2),
#             nn.ReLU(),
#             Flatten()
#         )
#
#         # self.fc1 = nn.Linear(h_dim, z_dim)
#         # self.fc2 = nn.Linear(h_dim, z_dim)
#         # self.fc3 = nn.Linear(z_dim, h_dim)
#         self.fc1 = nn.Linear(36864, z_dim)
#         self.fc2 = nn.Linear(36864, z_dim)
#         self.fc3 = nn.Linear(z_dim, 36864)
#
#         self.decoder = nn.Sequential(
#             # UnFlatten(),
#             # nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
#             nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
#             nn.Sigmoid(),
#         )
#
#     def reparameterize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         # return torch.normal(mu, std)
#         esp = torch.randn(*mu.size())
#         # print(std.device, esp.device)
#         esp = esp.to('cuda')
#         z = mu + std * esp
#         return z
#
#     def bottleneck(self, h):
#         mu, logvar = self.fc1(h), self.fc2(h)
#         z = self.reparameterize(mu, logvar)
#         return z, mu, logvar
#
#     def representation(self, x):
#         return self.bottleneck(self.encoder(x))[0]
#
#     def forward(self, x):
#         # h = self.encoder(x)
#         # print(self.encoder)
#         # print(x.shape)
#         B, C, H, W = x.shape
#         h = self.encoder[0](x)
#         h = self.encoder[1](h)
#         h = self.encoder[2](h)
#         h = self.encoder[3](h)
#         h = self.encoder[4](h)
#         h = self.encoder[5](h)
#         h = self.encoder[6](h)
#         h = self.encoder[7](h)
#         # print(h.shape)
#         h = self.encoder[8](h)
#         # print(h.shape)
#
#         z, mu, logvar = self.bottleneck(h)
#         z = self.fc3(z)
#         # print(z.shape)
#         # z = self.decoder(z)
#         z = z.view(B, 256, 12, 12)
#         z = self.decoder[0](z)
#         z = self.decoder[1](z)
#         z = self.decoder[2](z)
#         z = self.decoder[3](z)
#         z = self.decoder[4](z)
#         z = self.decoder[5](z)
#         z = self.decoder[6](z)
#         z = self.decoder[7](z)
#         # z = self.decoder[8](z)
#
#         return z, mu, logvar

class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)


class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))


class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()
        self.m1 = DecoderModule(256, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)


class VAE(nn.Module):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        # # latent features
        self.n_latent_features = 64

        # resolution
        # mnist, fashion-mnist : 28 -> 14 -> 7
        # cifar : 32 -> 8 -> 4
        # stl : 96 -> 24 -> 6
        # if dataset in ["mnist", "fashion-mnist"]:
        #     pooling_kernel = [2, 2]
        #     encoder_output_size = 7
        # elif dataset == "cifar":
        #     pooling_kernel = [4, 2]
        #     encoder_output_size = 4
        # elif dataset == "stl":
        #     pooling_kernel = [4, 4]
        #     encoder_output_size = 6

        pooling_kernel = [4, 4]
        encoder_output_size = 14

        color_channels = 3

        # # neurons int middle layer
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)


    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def _bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def sampling(self):
        # assume latent features space ~ N(0, 1)
        z = torch.randn(64, self.n_latent_features).to(self.device)
        z = self.fc3(z)
        # decode
        return self.decoder(z)

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        # decoder
        z = self.fc3(z)
        d = self.decoder(z)
        return d, mu, logvar


model = VAE().to(device)  # モデルをインスタンス化し、GPUにのせる
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # オプティマイザーの設定


# 損失関数
def loss_function(recon_x, x, mu, log_var):
    reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    vae_loss = reconstruction_loss + kl_loss
    return vae_loss, reconstruction_loss, kl_loss


losses = []
model.train()  # モデルを訓練モードに
epochs = 10
for epoch in range(epochs):
    train_loss = 0
    for i, (x, labels) in enumerate(train_loader):
        # 予測
        # x = x.to(device).view(-1, 32*32).to(torch.float32)
        x = x.to(device).to(torch.float32)
        # x_recon, mu, log_var, z = model(x)
        x_recon, mu, log_var = model(x)
        # 損失関数の計算
        loss, recon_loss, kl_loss = loss_function(x_recon, x, mu, log_var)

        # パラメータの更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 損失の表示
        if (i + 1) % 1 == 0:
            print(
                f'Epoch: {epoch + 1}, loss: {loss: 0.4f}, reconstruct loss: {recon_loss: 0.4f}, KL loss: {kl_loss: 0.4f}')
        losses.append(loss)
