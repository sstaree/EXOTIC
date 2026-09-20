import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear

# 内容编码器
class encoder(nn.Module):
    def __init__(self, n_dim,dims, n_z):
        super(encoder, self).__init__()
        self.enc_1 = Linear(n_dim, dims[0])
        self.enc_2 = Linear(dims[0], 2*dims[1])
        self.enc_3 = Linear(2*dims[1], 2*dims[2])
        self.z_layer = Linear(2*dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        enc_h4 = self.z_layer(enc_h3)
        z = self.z_b0(enc_h4)
        # z = F.normalize(z)
        return z

# class encoder(nn.Module):
#     def __init__(self, n_dim, dims, n_z):
#         super(encoder, self).__init__()
#         self.enc_1 = Linear(n_dim, dims[0])
#         self.enc_2 = Linear(dims[0], dims[1])
#         self.enc_3 = Linear(dims[1], dims[2])
#         self.z_layer = Linear(dims[2], n_z)
#         self.z_b0 = nn.BatchNorm1d(n_z,affine=False)
#
#     def forward(self, x):
#         enc_h1 = F.relu(self.enc_1(x))
#         enc_h2 = F.relu(self.enc_2(enc_h1))
#         enc_h3 = F.relu(self.enc_3(enc_h2))
#         z = self.z_b0(self.z_layer(enc_h3))
#         return z


# 解码器，将内容和风格的表示合并来重建输入
class Decoder(nn.Module):
    def __init__(self, n_dim, n_z,dims):
        super(Decoder, self).__init__()
        self.dec_0 = nn.Linear(n_z, n_z)
        self.dec_1 = nn.Linear(n_z, dims[2])
        self.dec_2 = nn.Linear(dims[2], dims[1])
        self.dec_3 = nn.Linear(dims[1], dims[0])
        self.x_bar_layer = nn.Linear(dims[0], n_dim)

    def forward(self, content, style):
        x = 0.5*(content+style)
        r = F.relu(self.dec_0(x))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar


# 解耦表示学习模型
class DecoupledRepresentationModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim ,dims):
        super(DecoupledRepresentationModel, self).__init__()
        self.content_encoder = encoder(input_dim, dims,latent_dim)
        # self.style_encoder = StyleEncoder(input_dim, latent_dim,dims)
        # self.content_encoder = ContentEncoder(input_dim,latent_dim)
        self.style_encoder = encoder(input_dim,dims,latent_dim)
        self.decoder = Decoder(input_dim, latent_dim,dims)

    def forward(self, x):
        content_rep = self.content_encoder(x)  # 获取内容表示
        style_rep = self.style_encoder(x)  # 获取风格表示
        reconstruction = self.decoder(content_rep, style_rep)  # 重建输入
        return reconstruction, content_rep, style_rep




# 示例训练过程
def train_model(model, data_loader, num_epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 使用均方误差作为重建损失

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            if isinstance(batch, list):
                batch = batch[0]  # 将 list 转换为 Tensor
            else:
                batch = batch
            inputs = batch
            optimizer.zero_grad()

            reconstruction, content_rep, style_rep = model(inputs)
            loss = criterion(reconstruction, inputs)  # 计算重建损失

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")


# 示例数据加载器
# from torch.utils.data import DataLoader, TensorDataset
#
# # 生成随机数据 (n_samples, input_dim)
# n_samples = 1000
# input_dim = 64
# latent_dim = 32
# output_dim = 64
# data = torch.randn(n_samples, input_dim)
#
# # 创建数据集和数据加载器
# dataset = TensorDataset(data)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# # 初始化并训练模型
# dims = [100,100,100]
# model = DecoupledRepresentationModel(input_dim, latent_dim, output_dim ,dims)
# train_model(model, data_loader)
