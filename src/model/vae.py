# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=300, latent_dim=15):
        super(VAE, self).__init__()
        
        # 编码器：将输入压缩到隐空间
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 均值 mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 对数方差 logvar
        
        # 解码器：从隐空间重建输入
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        # 重参数化技巧：z = mu + eps * std
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
