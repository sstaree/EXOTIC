import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from  DRL_model import DecoupledRepresentationModel
class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = Linear(n_dim, dims[0])
        # self.bn1 = nn.BatchNorm1d(dims[0])
        self.enc_2 = Linear(dims[0], dims[1])
        # self.bn2 = nn.BatchNorm1d(dims[1])
        self.enc_3 = Linear(dims[1], dims[2])
        # self.bn3 = nn.BatchNorm1d(dims[2])
        self.z_layer = Linear(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_b0(self.z_layer(enc_h3))
        return z


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

from torch_geometric.nn import GCNConv
class AE(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(AE, self).__init__()
        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)
        # self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input)-1)])
        # self.decoder_list = nn.ModuleList([decoder(n_input[i], dims[i], n_z) for i in range(len(n_input)-1)])

        self.encoder_list = nn.ModuleList([DecoupledRepresentationModel(n_input[i], n_z  ,n_input[i], dims[i]) for i in range(len(n_input) - 1)])
        # self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z )  for i in range(len(n_input) - 1)])
        self.decoder_list = nn.ModuleList([decoder(n_input[i], dims[i], n_z ) for i in range(len(n_input) - 1)])

        self.encoder_word = encoder(n_input[-1],dims[-1],n_z)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_z, nhead=1, dim_feedforward=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gates = nn.Parameter(torch.ones(len(n_input)-1, n_z))
        self.regression = Linear(n_z, nLabel)
        self.gnn = GCNConv(n_z, n_z)
        self.act = nn.Sigmoid()

    def forward(self, mul_X, word_embedding_train_batch,we,word_embedding_train):

        # enc0_h1 = F.relu(self.enc0_1(x0))
        # enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        # enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        # z0 = self.z0_b0(self.z0_layer(enc0_h3))

        summ = 0
        individual_zs = []
        individual_zs_imputation = []
        z_word = self.encoder_word(word_embedding_train)
        z_word = F.normalize(z_word)
        x_bar_list = []
        content_style_pair = []
        for enc_i, enc in enumerate(self.encoder_list):
            reconstruction,z_i,z_style_i = enc(mul_X[enc_i])
            individual_zs.append(z_i)
            content_style_pair.append([z_i,z_style_i])
            x_bar_list.append(reconstruction)
            miss_index = torch.diag(1-we[:,enc_i])
            imputation_value = miss_index.mm(z_word)
            z_i = torch.diag(we[:, enc_i]).mm(z_i)
            z_i = imputation_value + z_i
            summ += z_i
            individual_zs_imputation.append(z_i)


        # for enc_i, enc in enumerate(self.encoder_list):
        #     z_i = enc(mul_X[enc_i])
        #     individual_zs.append(z_i)
        #     miss_index = torch.diag(1-we[:,enc_i])
        #     imputation_value = miss_index.mm(z_word)
        #     z_i = torch.diag(we[:, enc_i]).mm(z_i)
        #     z_i = imputation_value + z_i
        #     summ += z_i
        #     individual_zs_imputation.append(z_i)

        #attention fusion
        # zs_transformer = self.transformer_encoder(torch.stack(individual_zs_imputation))
        # z = zs_transformer.mean(dim=0)

        #weighted fusion
        wei = 1 / we.shape[1]
        z = wei * summ #融合后的z

        #weighted fusioin(matrix)
        # z = torch.stack(individual_zs_imputation).mean(0)  # 融合后的z

        #gate fusion
        # weighted_views = [individual_zs_imputation[i] * self.gates[i] for i in range(len(individual_zs_imputation))]
        # z = torch.sum(torch.stack(weighted_views), dim=0)


        # # decoder0
        # r0 = F.relu(self.dec0_0(z))
        # dec0_h1 = F.relu(self.dec0_1(r0))
        # dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        # dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        # x0_bar = self.x0_bar_layer(dec0_h3)

        x_bar_list = []
        for dec_i, dec in enumerate(self.decoder_list):
            x_bar_list.append(dec(individual_zs[dec_i]))
        yLable = self.regression(F.relu(z))
        return x_bar_list, yLable, z, individual_zs ,z_word , content_style_pair


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

    def forward(self, mul_X,word_embedding_train_batch, we,word_embedding_train):
        x_bar_list, target_pre, fusion_z, individual_zs ,z_word ,content_style_pair = self.ae(mul_X,word_embedding_train_batch, we ,word_embedding_train)

        return x_bar_list, target_pre, fusion_z, individual_zs ,z_word ,content_style_pair

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    
    
def get_model(d_list,n_layers=4,classes_num=10,device=torch.device('cuda:0')):
    

    model = DICNet(n_stacks=n_layers,n_input=d_list,n_z=classes_num,Nlabel=classes_num).to(device)
    model = model.to(device)
    
    return model