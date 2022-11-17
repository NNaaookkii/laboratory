# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch画像用
import torchvision
import torchvision.transforms as transforms

# 画像表示用
import matplotlib.pyplot as plt


from utils.dataloader import get_loader
batch_size = 32
train_size = 352
train_path = './dataset/sekkai_TrainDataset'
val_path = './dataset/sekkai_ValDataset'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataloaderの作成
image_root = '{}/images/'.format(train_path)
gt_root = '{}/masks/'.format(train_path)

image_root_val = '{}/images/'.format(val_path)
gt_root_val = '{}/masks/'.format(val_path)

train_loader = get_loader(image_root, gt_root, batchsize=batch_size, trainsize=train_size)
val_loader = get_loader(image_root_val, gt_root_val, batchsize=batch_size, trainsize=train_size, phase='val')


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # ニューラルネットワークで事後分布の平均・分散を計算する
        h = torch.relu(self.fc(x))
        mu = self.fc_mu(h) # μ
        log_var = self.fc_var(h) # log σ^2
        
        # 潜在変数を求める
        ## 標準正規乱数を振る
        eps = torch.randn_like(torch.exp(log_var))
        ## 潜在変数の計算 μ + σ・ε
        z = mu + torch.exp(log_var/2)*eps
        return mu, log_var, z
    

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, z):
        h = torch.relu(self.fc(z))
        output = torch.sigmoid(self.fc_output(h))
        return output
    
# VAEのモデル作成
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)
        
    def forward(self, x):
        mu, log_var, z = self.encoder(x) # エンコード
        x_decoded = self.decoder(z) # デコード
        return x_decoded, mu, log_var, z
    
# 損失関数
def loss_function(label, predict, mu, log_var):
    reconstruction_loss = F.binary_cross_entropy(predict, label, reduction='sum')
    kl_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    vae_loss = reconstruction_loss + kl_loss
    return vae_loss, reconstruction_loss, kl_loss
    
# ハイパーパラメータ
image_size = 32 * 32
h_dim = 32
z_dim = 16
num_epochs = 10
learning_rate = 1e-3

model = VAE(image_size, h_dim, z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 学習
losses = []
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for i, (x, labels) in enumerate(train_loader):
        # 予測
        x = x.to(device).view(-1, image_size).to(torch.float32)
        x_recon, mu, log_var, z = model(x)
        # 損失関数の計算
        loss, recon_loss, kl_loss = loss_function(x, x_recon, mu, log_var)
        
        # パラメータの更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 損失の表示
        if (i+1) % 10 == 0:
            print(f'Epoch: {epoch+1}, loss: {loss: 0.4f}, reconstruct loss: {recon_loss: 0.4f}, KL loss: {kl_loss: 0.4f}')
        losses.append(loss)    
        
# 画像の生成
model.eval()

with torch.no_grad():
    z = torch.randn(25, z_dim).to(device)
    out = model.decoder(z)
out = out.view(-1, 32, 32)
out = out.cpu().detach().numpy()

# 画像の表示
fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
plt.gray()
for i in range(25):
    idx = divmod(i, 5)
    ax[idx].imshow(out[i])
    ax[idx].axis('off');
fig.savefig("fig_VAE/VAE_1.png")




