import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.nn import Linear
from  DRL_model import DecoupledRepresentationModel,encoder
import numpy as np
from typing import Any, Optional, Tuple
from measure import *
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from torch.distributions.multivariate_normal import MultivariateNormal

class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]

        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradientReversalLayer.apply(x, coeff)
class Measure_F(nn.Module):
    def __init__(self, view1_dim, view2_dim, phi_size, psi_size, latent_dim=1):
        super(Measure_F, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.psi = MLP(view2_dim, psi_size, latent_dim)
        # gradient reversal layer
        self.grl1 = GradientReversalLayer()
        self.grl2 = GradientReversalLayer()

    def forward(self, x1, x2):
        y1 = self.phi(grad_reverse(x1,1))
        y2 = self.psi(grad_reverse(x2,1))
        return y1, y2

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


class decoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder, self).__init__()
        self.dec_0 = Linear(n_z, n_z)
        self.dec_1 = Linear(n_z, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar


class LinearAlign(nn.Module):
    def __init__(self,d1,d2,d_shared):
        super(LinearAlign, self).__init__()
        # 定义映射矩阵 W_image 和 W_text
        self.W_image = nn.Parameter(torch.randn(d1, d_shared))
        self.W_text = nn.Parameter(torch.randn(d2, d_shared))

    def forward(self,X_image,X_text):
        X_image_aligned = torch.matmul(X_image, self.W_image)  # 对图像特征进行线性变换
        X_text_aligned = torch.matmul(X_text, self.W_text)    # 对文本特征进行线性变换
        return X_image_aligned, X_text_aligned


class AE(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(AE, self).__init__()
        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                # linshidim = round(n_dim * 0.8)
                linshidim = round(1024)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)

        # self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input)-1)])
        # self.decoder_list = nn.ModuleList([decoder(n_input[i], dims[i], n_z) for i in range(len(n_input)-1)])

        self.encoder_list = nn.ModuleList([DecoupledRepresentationModel(n_input[i], n_z  ,n_input[i], dims[i]) for i in range(len(n_input) - 1)])
        # self.encoder_word_list = nn.ModuleList([encoder(n_z,[1024,1024,1500],n_z) for i in range(len(n_input) - 1)])
        # self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z )  for i in range(len(n_input) - 1)])
        # self.decoder_list = nn.ModuleList([decoder(n_input[i], dims[i], n_z ) for i in range(len(n_input) - 1)])

        # self.encoder_word = encoder(n_input[-1],dims[-1],n_z)
        # self.encoder_word1 = encoder(n_z,[1024,1024,1500],n_z)
        # self.encoder_word1 = Linear(n_z,nLabel)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=n_z, nhead=2, dim_feedforward=2)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # encoder_layer1 = nn.TransformerEncoderLayer(d_model=n_z, nhead=2, dim_feedforward=2)
        # self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer1, num_layers=6)
        # self.gates = nn.Parameter(torch.ones(2, ))
        self.regression = Linear(n_z, nLabel)  
        # self.regression = encoder(n_z,[1024,1024,1500],nLabel)
        # self.regression_z1 = encoder(n_z,[1024,1024,1500],n_z)
        # self.regression_word = Linear(n_z, nLabel)    # 20250306 23.59前用

        # self.regression1=Linear(n_z,nLabel)
        # self.regression_copy = Linear(n_z, nLabel)

        # self.select_index = Linear(nLabel,3200)
        self.nLabel = nLabel
        self.emb_w_ini = encoder(n_z,[1024,1024,1500],n_z)
        # self.emb_w_ini = Linear(n_z, n_z)
        # self.emb_second_phase = encoder(n_z,[1024,1024,1500],n_z)
        # self.emb_w = encoder(n_z,[1024,1024,1500],n_z)


        # 设置新的线性层的参数为不可训练状态
        # for param in self.regression.parameters():
        #     param.requires_grad = False
        # self.act = nn.Sigmoid()
        self.prev_result = None
        self.nLabel = nLabel




    # 假设 H 是一个 (M, D) 的张量，M 是样本数，D 是特征维度
    # 假设 Z 是一个 (N, D) 的张量，N 是样本数，D 是特征维度

    # 高斯建模
    def gaussian_modeling(self,H):
        mean = torch.mean(H, dim=0)  # 计算均值，形状为 (D,)
        cov = torch.cov(H.T)         # 计算协方差矩阵，形状为 (D, D)
        return mean, cov

    # 计算相似度（使用马氏距离）
    def compute_similarity(self,z, mean, cov):
        # 创建多元高斯分布
        mvn = MultivariateNormal(mean, covariance_matrix=cov)
        # 计算对数概率密度（马氏距离与对数概率密度相关）
        log_prob = mvn.log_prob(z)  # z 可以是单个样本或批量样本
        return -log_prob  # 返回负对数概率密度（越小表示越相似）

    # 为 Z 中的每个表示匹配外部知识
    def match_external_knowledge(self,Z, H):
        mean, cov = self.gaussian_modeling(H)
        matched_indices = []
        
        for z in Z:
            # 计算 Z 中每个样本与 H 的高斯分布的相似度
            similarity = self.compute_similarity(z, mean, cov)
            matched_indices.append(similarity.argmin().item())  # 找到最相似的样本索引
        
        return matched_indices




    def forward(self, mul_X,we,word_embedding_train,k,epoch,mode):

        summ = 0
        summ1 = 0
        summ2 = 0
        summ3 = 0
        individual_zs = []
        individual_zs_p = []
        individual_zs_imputation = []

        z_word = word_embedding_train
        z_word = self.emb_w_ini(z_word)

        # z_word = z_word / z_word.norm(dim=1, keepdim=True)    # 新
  

        content_style_pair = []
        z_word_list = []


        for enc_i, enc in enumerate(self.encoder_list):
            _, z_i, z_style_i = enc(mul_X[enc_i])
            content_style_pair.append([z_i, z_style_i])
            individual_zs.append(z_i)
            individual_zs_p.append(z_style_i)
            # summ1 += torch.diag(we[:, enc_i]).mm(z_i)
            # summ2 += torch.diag(we[:, enc_i]).mm(z_style_i)
            summ1 += (we[:, enc_i].unsqueeze(1) * z_i)
            summ2 += (we[:, enc_i].unsqueeze(1) * z_style_i)
        wei = 1 / torch.sum(we, 1)

        # z1 = torch.diag(wei).mm(summ1)  # 第一个通道的融合表示，也是用来匹配筛选外部知识的表示
        z1 = (wei.unsqueeze(1) * summ1)
        # z2 = torch.diag(wei).mm(summ2)



        dist = torch.mm(z1, z_word.t())
        _, neighbors = dist.topk(k, dim=1, largest=True, sorted=True)
        neighbors = neighbors.view(-1)
        z_word_select1 = z_word[neighbors]
        z_word_select = z_word_select1.view(int(neighbors.shape[0] / k), k, z_word.shape[1]).mean(dim=1)


        z_word_select_all = z_word_select1
        # print(z_word_select_all.shape[0])

        

        summ4 = 0

        for enc_i, enc in enumerate(self.encoder_list):
            z_i = individual_zs[enc_i]
            z_i_p = individual_zs_p[enc_i]
            miss_index = torch.diag(1-we[:,enc_i])
            imputation_value = miss_index.mm(z_word_select)     #外部知识
            # imputation_value_0 = torch.zeros(imputation_value.shape[0],imputation_value.shape[1]).cuda()
            # imputation_value_mean = miss_index.mm(z1)   #z2用平均表示补全，z1用第一个通道的平均表示补全
            imputation_value = imputation_value

            # z_i = torch.diag(we[:, enc_i]).mm(z_i)
            # z_i_p = torch.diag(we[:, enc_i]).mm(z_i_p)

            z_i += (we[:, enc_i].unsqueeze(1) * z_i)
            z_i_p = (we[:, enc_i].unsqueeze(1) * z_i_p)

            summ4 += z_i_p
            z_i = imputation_value + z_i
            z_i_p = imputation_value + z_i_p      #分类通道，这个是论文用的

            summ += z_i
            summ3 += z_i_p
            individual_zs_imputation.append(z_i)

        wei = 1 / we.shape[1]
        # z = wei * summ

        # summ4 = torch.diag(1 / torch.sum(we, 1)).mm(summ4)      #不补得融合表示

        z_d = (1 / we.shape[1]) * summ3     #补的融合表示

        # x_bar_list = []
        # for dec_i, dec in enumerate(self.decoder_list):
        #     x_bar_list.append(dec(individual_zs[dec_i]))

        # yLable = self.regression(F.relu(z1))[:,0:self.nLabel]
        # yLable_word = self.regression_word(F.relu(z_word_select))[:,0:self.nLabel]

        yLable = self.regression(F.relu(z_d))
        # yLable_word = self.regression_word(F.relu(z_word_select))
        # yLable_word = self.regression_word(F.relu(self.emb_second_phase(z_word_select)))



        return 1,yLable,1,z1,individual_zs_imputation,z_word_select,individual_zs_p,z_word_list,z_d,z_word_select_all






# class AE(nn.Module):
#     def __init__(self, n_stacks, n_input, n_z, nLabel, k=5):
#         super(AE, self).__init__()
#         self.k = k  # 近邻数
#         dims = []
#         for n_dim in n_input:
#             linshidims = []
#             for idim in range(n_stacks - 2):
#                 linshidim = round(1024)
#                 linshidim = int(linshidim)
#                 linshidims.append(linshidim)
#             linshidims.append(1500)
#             dims.append(linshidims)

#         self.encoder_list = nn.ModuleList([DecoupledRepresentationModel(n_input[i], n_z, n_input[i], dims[i]) 
#                                            for i in range(len(n_input) - 1)])
#         self.regression = nn.Linear(n_z, nLabel)
#         self.nLabel = nLabel
#         self.emb_w_ini = encoder(n_z, [1024, 1024, 1500], n_z)

#     def forward(self, mul_X, we, word_embedding_train, k, epoch, mode):
#         summ = 0
#         summ3 = 0
#         individual_zs = []
#         individual_zs_p = []
#         individual_zs_imputation = []
#         k = 1

#         # 对词嵌入进行初始编码
#         z_word = self.emb_w_ini(word_embedding_train)

#         content_style_pair = []

#         # 1️⃣ 对每个视图进行编码
#         for enc_i, enc in enumerate(self.encoder_list):
#             _, z_i, z_style_i = enc(mul_X[enc_i])
#             content_style_pair.append([z_i, z_style_i])
#             individual_zs.append(z_i)
#             individual_zs_p.append(z_style_i)

#         n_samples = we.shape[0]
#         n_views = we.shape[1]

#         # 2️⃣ 对每个视图缺失的样本，用k邻居补全
#         for enc_i in range(n_views):
#             z_i = individual_zs[enc_i]
#             z_i_p = individual_zs_p[enc_i]
            
#             miss_mask = 1 - we[:, enc_i]  # 1 表示缺失，0 表示存在
#             miss_indices = torch.nonzero(miss_mask).squeeze(1)
#             exist_indices = torch.nonzero(we[:, enc_i]).squeeze(1)

#             if len(miss_indices) > 0 and len(exist_indices) > 0:
#                 # 计算相似度（用余弦相似度）
#                 z_exist = z_i[exist_indices]
#                 z_exist_norm = F.normalize(z_exist, dim=1)
                
#                 for idx in miss_indices:
#                     z_miss = z_i[idx].unsqueeze(0)  # 当前缺失样本
#                     # 使用现有样本计算相似度
#                     sim = torch.mm(F.normalize(z_miss, dim=1), z_exist_norm.t())  # (1, n_exist)
#                     topk_sim, topk_idx = sim.topk(min(k, sim.shape[1]), dim=1)
#                     # 平均 k 邻居的表示补全
#                     z_i[idx] = z_exist[topk_idx.squeeze(0)].mean(dim=0)
#                     z_i_p[idx] = z_i[idx]  # 分类通道也用同样补全

#             # 融合存在样本的加权表示
#             z_i = we[:, enc_i].unsqueeze(1) * z_i
#             z_i_p = we[:, enc_i].unsqueeze(1) * z_i_p

#             summ += z_i
#             summ3 += z_i_p
#             individual_zs_imputation.append(z_i)

#         # 3️⃣ 最终融合表示
#         z_d = (1 / n_views) * summ3

#         # 4️⃣ 分类输出
#         yLable = self.regression(F.relu(z_d))

#         return 1, yLable, 1, None, individual_zs_imputation, None, individual_zs_p, [], z_d, None





class AE_word(nn.Module):

    def __init__(self, n_feature, n_z, nLabel):
        super(AE_word, self).__init__()
        
        self.encoder = encoder(n_feature,[1024,1024,1500],n_z)

        self.regression = Linear(n_z, nLabel)

    def forward(self, word):
        z = self.encoder(word)

        logits = self.regression(z)
        

        return z,logits


class DICNet(nn.Module):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 Nlabel):
        super(DICNet, self).__init__()

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z,
            nLabel=Nlabel)

    def forward(self, mul_X, we, word_embedding_train,k,epoch,mode='train'):
        x_bar_list, target_pre,yLable_word, fusion_z, individual_zs ,z_word ,content_style_pair,view_neighbor,z_word_label,z_word_select_all = self.ae(mul_X, we ,word_embedding_train,k,epoch,mode)

        return x_bar_list, target_pre,yLable_word, fusion_z, individual_zs ,z_word ,content_style_pair,view_neighbor,z_word_label,z_word_select_all

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)



# def get_model(d_list,n_layers=4,classes_num=10,device=torch.device('cuda:0')):


#     model = DICNet(n_stacks=n_layers,n_input=d_list,n_z=classes_num,Nlabel=classes_num).to(device)
#     model = model.to(device)

#     return model