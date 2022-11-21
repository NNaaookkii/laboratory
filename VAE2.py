#ライブラリの準備
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils as utils
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils.dataloader import get_loader
batch_size = 32
train_size = 352
train_path = './dataset/sekkai_TrainDataset'
val_path = './dataset/sekkai_ValDataset'

# モデルの設計
class VAE(nn.Module):
    def __init__(self, z_dim, x_dim=28*28):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        # エンコーダ用の数数
        self.fc1 = nn.Linear(x_dim, 20)
        self.fc2_mean = nn.Linear(20, z_dim)
        self.fc2_var = nn.Linear(20, z_dim)
        # デコーダ用の関数
        self.fc3 = nn.Linear(z_dim, 20)
        self.fc4 = nn.Linear(20, x_dim)
    # エンコーダ
    def encoder(self, x):
        x = x.view(-1, self.x_dim)
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x) # 平均
        log_var = self.fc2_var(x) # 分散の対数
        return mean, log_var
    # 潜在ベクトルのサンプリング（再パラメータ化）
    def reparametrizaion(self, mean, log_var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon*torch.exp(0.5 * log_var)
    # デコーダ
    def decoder(self, z):
        y = F.relu(self.fc3(z))
        y = torch.sigmoid(self.fc4(y)) # 各要素にシグモイド関数を適用し、値を(0,1)の範囲に
        return y
    def forward(self, x, device):
        x = x.view(self, x_dim)
        mean, log_var = self.encoder(x) # 画像xを入力して、平均・分散を出力
        KL = 0.5 * torch.sum(1+log_var - mean**2 - torch.exp(log_var)) # KL[q(z|x)||p(z)]を計算
        z = self.reparametrizaion(mean, log_var, device) # 潜在ベクトルをサンプリング(再パラメータ化)
        x_hat = self.decoder(z) # 潜在ベクトルを入力して、再構築画像yを出力
        reconstruction = torch.sum(x * torch.log(x_hat+1e-8) + (1-x) * torch.log(1 - x_hat + 1e-8)) # E[log p(x|z)]
        lower_bound = -(KL + reconstruction) # 変分下限(ELBO)=E[log p(x|z) - KL[q(z|x)||p(z)]]
        return lower_bound, z, x_hat

# dataloaderの作成
image_root = '{}/images/'.format(train_path)
gt_root = '{}/masks/'.format(train_path)
image_root_val = '{}/images/'.format(val_path)
gt_root_val = '{}/masks/'.format(val_path)
train_loader = get_loader(image_root, gt_root, batchsize=batch_size, trainsize=train_size)
val_loader = get_loader(image_root_val, gt_root_val, batchsize=batch_size, trainsize=train_size, phase='val')

# 学習
model = VAE(z_dim = 10).to(device) # モデルをインスタンス化し、GPUにのせる
optimizer = optim.Adam(model.parameters(), lr=1e-3) # オプティマイザーの設定
model.train() # モデルを訓練モードに
num_epochs = 10
loss_list = []
for i in range(num_epochs):
    losses = []
    for x, t in train_loader: # データローダーからデータを取り出す
        x = x.to(device) # データをGPUにのせる
        loss, z, y = model(x, device) # 損失関数の値 loss, 潜在ベクトル z, 再構築画像 yを入力
        model.zero_grad() # モデルの勾配を初期化
        loss.backward() # モデル内のパラメータの勾配を計算
        optimizer.step() # 最適化を実行
        losses.append(loss.cpu().detach().numpy()) # ミニバッチの損失を記録
    loss_list.append(np.average(losses)) # バッチ全体の損失を登録
    print("EPOCH: {} loss: {}".format(i, np.average(losses)))

# 可視化
fig = plt.figure(figsize=(20,4))
model.eval()
zs = []
for x, t in dataloader_valid:
    for i, im in enumerate(x.view(-1,28,28).detach().numpy()[:10]):
        # 元画像を可視化
        ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
        ax.imshow(im, "gray")
    x = x.to(device)
    _, _, y = model(x, device) #再構築画像 y を出力
    y  = y.view(-1,28,28)
    for i, im in enumerate(y.cpu().detach().numpy()[:10]):
        # 再構築画像を可視化
        ax = fig.add_subplot(2,10,11+i, xticks=[], yticks=[])
        ax.imshow(im, "gray")