# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import random
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from model import DICNet,AE_word
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import scipy.io
from sklearn.preprocessing import OneHotEncoder
import h5py
import math
from loss import Loss
import pickle
import matplotlib.pyplot as plt
# import os
import pandas as pd
import csv
import torch.optim as optim
import os
from sklearn.cluster import KMeans
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from measure import *
parser = argparse.ArgumentParser()

# parser.add_argument('--lrkl', type=float, default=0.05)
# parser.add_argument('--lrkl1', type=float, default=0.001)
parser.add_argument('--lrkl', type=float, default=0.05)   #caltech101-20 5e-3;scene15：5e-4
parser.add_argument('--lrkl1', type=float, default=0.001)
parser.add_argument('--Nlabel', default=7, type=int)
parser.add_argument('--Nz', default=7, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--warm_epoch', default=20,type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--MaskRatios', type=float, default=0.5)
parser.add_argument('--LabelMaskRatio', type=float, default=0.8)
parser.add_argument('--TraindataRatio', type=float, default=0.8)
parser.add_argument('--AE_shuffle', type=bool, default=True)
parser.add_argument('--min_AP', default=0., type=float)
parser.add_argument('--tol', default=1e-7, type=float)
parser.add_argument('--noiseRatio', default=0, type=float)

parser.add_argument('--miss_rate', default=0, type=float)
parser.add_argument('--lambda_epochs', type=int, help='gradually increase the value of lambda from 0 to 1', default=100)

parser.add_argument('--dataset', type=str, help='PIE, Caltech101, UCI, BBC, Leaves Scene LandUse_21',
                    default="PIE")  # Due to file size limitations, we have only uploaded the PIE, Leaves, and UCI datasets here
parser.add_argument('--noise_type', type=str, help="sym, flip, IDN", default="IDN")
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--tau', type=float, default=0.5)
parser.add_argument('--q', type=float, default=0.1)
parser.add_argument('--momentumkl', type=float, default=0.5)
parser.add_argument('--knn_threshold', type=float, default=0)
parser.add_argument('--high_scor', default=0.95, type=float)
parser.add_argument('--num_class', default=1.0, type=float)
parser.add_argument('--correct', default=True, type=bool)
parser.add_argument('--imputation', default=True, type=bool)
parser.add_argument('--k', default=5, type=int)

args = parser.parse_args()

args.data_path = './Datasets/' + args.dataset


class My_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, zp_pro_0, p0):
        r = 0.5
        sf = nn.Softmax(dim=1)
        zp_pro_0 = zp_pro_0.float()
        p0 = p0.float()
        zp_pro_0 = sf(zp_pro_0)  # 模型预测的zp_pro_0，转化为概率
        res = torch.mm(zp_pro_0, p0.t())  #
        res = res.diag()
        res = ((1 - res ** r) / r).mean()
        return res


class My_BCE_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, zp_pro_0, p0):
        r = 0.5
        sf = nn.Softmax(dim=1)
        zp_pro_0 = zp_pro_0.float()
        p0 = p0.float()
        zp_pro_0 = sf(zp_pro_0)  # 模型预测的zp_pro_0，转化为概率
        res = torch.mm(zp_pro_0, p0.t())  #
        res = res.diag()
        res = res.mean()
        return res

def get_mask(view_num, alldata_len, missing_rate):

    missindex = np.ones((alldata_len, view_num))
    b=((10 - 10*missing_rate)/10) * alldata_len
    miss_begin = int(b)
    for i in range(miss_begin, alldata_len):
        missdata = np.random.randint(0, high=view_num,
                                     size=view_num - 1)

        missindex[i, missdata] = 0

    return missindex


def set_seed(seed=2000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mseLoss(pred, target, model):
    loss = nn.CrossEntropyLoss(reduction='mean')
    regularization_loss = 0  # 正则项
    for param in model.parameters():
        regularization_loss += torch.sum(param ** 2)  # 计算所有参数平方
    return loss(pred, target) + 0.01 * regularization_loss  # 返回损失。


def newLoss(f, s):
    r = 0.5
    return (1. - (s * f).sum(1) ** r) / r





# loss = nn.CrossEntropyLoss()
loss = mseLoss
loss1 = My_loss()
loss2 = My_BCE_loss()


def train_DIC(mul_X, mul_X_val, WE, WE_val, yt_label_noisy, yv_label, yt_label_clean, yv_label_clean,word_embedding_train, device, args):
    # return None, torch.randn(9, 1)
    model = torch.load(r"D:\python project\MVMLC\新方法 - 副本\UCI(0.9)_model.pth")
    model1 = AE_word(512,args.N_z,args.Nlabel).to(device)
    loss_model = Loss(args.alpha, device)
    num_label = yv_label.shape[1]
    args.num_class = num_label
    yt_label_correct = yt_label_noisy.copy()
    mul_X_1 = mul_X
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Module):
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    nn.init.constant_(mm.bias, 0.0)

    num_X = mul_X[0].shape[0]  # 样本数量
    optimizer = SGD(model.parameters(), lr=0.1, momentum=args.momentumkl)
    # optimizer = Adam(model.parameters(), lr=args.lrkl)
    plt_val_acc = []
    plt_train_acc = []
    plt_val_loss = []
    model.eval()
    word_embedding_train = word_embedding_train.to(device)
    yt_label_clean = torch.tensor(yt_label_clean).to(device)
    yv_label_clean = torch.tensor(yv_label_clean).to(device)
    mul_X_1 = [item.to(device) for item in mul_X]
    mul_X_1_test = [item.to(device) for item in mul_X_val]
    x_bar_list, yLable_logit,yLable_word_logit, fusion_z, individual_zs, z_word ,content_style_pair,z_word_list,prompts_index  = model(mul_X_1,WE.to(device),word_embedding_train,args.k,0,'train')
    x_bar_list_test, yLable_logit_test,yLable_word_logit_test, fusion_z_test, individual_zs_test, z_word_test ,content_style_pair_test,z_word_list_test,prompts_index_test  = model(mul_X_1_test,WE_val.to(device),word_embedding_train,args.k,0,'val')
    # prompts = word_embedding_train[prompts_index]
    prompts = fusion_z.detach()
    prompts_test = fusion_z_test.detach()
    second_phase(prompts,yt_label_clean,prompts_test,yv_label_clean)

    exit()
    for epoch in range(int(args.epochs)):
        model.train()
        index_array = np.arange(num_X)
        if args.AE_shuffle == True:
            np.random.shuffle(index_array)


        train_loss,train_acc_fea,train_acc_word,cont_loss,dis_loss = warmup(epoch,mul_X_1,yt_label_clean,yt_label_correct,word_embedding_train,args,model,device,optimizer,loss_model,loss,WE,index_array)
        if (epoch+1)%10==0:
            val_loss, val_acc_fea,val_acc_word,fusion_z_val = test(epoch, mul_X_val, yv_label_clean, yv_label, args, model, device, optimizer, loss_model,loss, WE_val, index_array, word_embedding_train)
            plt_val_loss.append(val_loss)
            plt_val_acc.append(val_acc_fea)
            plt_train_acc.append(train_acc_fea)
            print(
                "epoch:{}    train_loss:{:.4f}   train_acc:{:.4f}       cont_loss:{:.2f}    dis_loss:{:.2f}     val_acc_fea:{:.4f}  val_acc_word:{:.4f}".format(epoch+1, train_loss,
                                                                                             train_acc_fea,cont_loss,dis_loss,
                                                                                             val_acc_fea,val_acc_word))
    torch.save(model, "D:\python project\MVMLC\新方法 - 副本\{}({})_model.pth".format(args.dataset,args.miss_rate))
    # with open('feature_{}_{}.pkl'.format(args.dataset,args.miss_rate), 'wb') as file:
    #     pickle.dump(fusion_z_val, file)
    # with open('label_{}_{}.pkl'.format(args.dataset, args.miss_rate), 'wb') as file:
    #     pickle.dump(yv_label_clean, file)

        # plt_train_loss.append(train_loss)
        # plt_train_acc.append(train_acc)

        # print("epoch:{}    train_loss:{:.4f}   train_acc:{:.4f}     val_acc:{:.4f}".format(epoch,plt_train_loss[-1],plt_train_acc[-1],plt_val_acc[-1]))
    # plt.plot(plt_val_acc)
    # plt.plot(plt_train_acc)
    # plt.show()
    return plt_train_acc[-1] * 100, plt_val_acc[-1] * 100

def second_phase(prompts,yt_label_clean,prompts_test,yv_label_clean):
    
    model = AE_word(512,args.N_z,args.Nlabel).cuda()
    optimizer = SGD(model.parameters(), lr=0.05, momentum=args.momentumkl)

    for epoch in range(500):
        model.train()
        z,logits  = model(prompts)
        loss_CL = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * yt_label_clean, dim=1))
        
        loss_CL.backward()
        optimizer.step()
        optimizer.zero_grad()
        right_num = np.sum(np.argmax(logits.detach().cpu().numpy(),axis=1)==np.argmax(yt_label_clean.detach().cpu().numpy(),axis=1))
        total_num = prompts.shape[0]
        print("epoch:{}     loss:{:.4f}      acc:{:.2f}".format(epoch,loss_CL.item(),(right_num/total_num)*100))

    model.eval()
    z,logits  = model(prompts_test)
    right_num = np.sum(np.argmax(logits.detach().cpu().numpy(),axis=1)==np.argmax(yv_label_clean.detach().cpu().numpy(),axis=1))
    total_num = prompts_test.shape[0]
    print("test acc:{:.2f}".format((right_num/total_num)*100))
import scipy.io as sio

def main():
    global args
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    X, Y_clean, Y_noisy , word_embedding = import_and_load_data(args.dataset)

    word_embedding = word_embedding[0:50000,:]

    # X = [X[2]]


    X_train, X_test, y_train, y_test = balanced_train_test_split(X, Y_clean)

    X_train_tensor = [torch.tensor(ivx).float() for ivx in X_train]
    X_test_tensor = [torch.tensor(ivx).float() for ivx in X_test]

    num_view = len(X)
    WE_train = get_mask(num_view, X_train[0].shape[0], args.miss_rate)
    WE_val = get_mask(num_view, X_test[0].shape[0], args.miss_rate)

    # for i in range(WE_train.shape[0]):
    #     for j in range(num_view):
    #         if WE_train[i][j]==0:
    #             mul_X_trian[j][i] = np.zeros(X[j].shape[1])
    # for i in range(WE_val.shape[0]):
    #     for j in range(num_view):
    #         if WE_val[i][j]==0:
    #             mul_X_val[j][i] = np.zeros(X[j].shape[1])
    #
    # WE_train = np.ones((train_index.shape[0],num_view))
    # WE_val = np.ones((val_index.shape[0],num_view))

    WE_train = torch.tensor(WE_train).float()
    WE_val = torch.tensor(WE_val).float()

    # word_embedding = np.random.randn(50000, 768)

    word_embedding_train = F.normalize(torch.tensor(word_embedding)).float()
    word_embedding_test = F.normalize(torch.tensor(word_embedding)).float()

    # word_embedding_train = F.normalize(torch.randn((50000,768)))
    # word_embedding_test = F.normalize(torch.randn((50000,768)))

    args.Nlabel = y_train.shape[1]
    args.N_z = word_embedding_train.shape[1]
    # args.N_z = args.Nlabel
    args.n_input = [xiv.shape[1] for xiv in X_train]
    args.n_input.append(word_embedding_train.shape[1])
    print(args.n_input)



    # print(WE_val)
    # print(np.argmax(y_test,axis=1))

    train_max_acc, val_max_acc = train_DIC(X_train_tensor, X_test_tensor, WE_train, WE_val, y_train, y_test, y_train,y_test, word_embedding_train,device, args)
    return train_max_acc, val_max_acc


# def main():
#     global args
#     args.cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if args.cuda else "cpu")
#
#     X, Y_clean, Y_noisy , word_embedding = import_and_load_data(args.dataset)
#
#     X_train, X_test, y_train, y_test = balanced_train_test_split(X,Y_clean)
#
#     view_num = len(X)
#
#     label = Y_noisy
#     label_clean = Y_clean
#     mul_X = [None] * view_num
#
#     indexstart = [k for k in range(label.shape[0])]
#     random.shuffle(indexstart)
#
#     folds_sample_index = np.array([indexstart])
#     random.shuffle(folds_sample_index)
#
#     Ndata, args.Nlabel = label.shape
#     args.N_z = args.Nlabel
#     args.N_z = 768
#     num_view = view_num
#     num_data = Ndata
#     indexperm = folds_sample_index
#
#     train_num = math.ceil(Ndata * args.TraindataRatio)
#
#     train_index = indexperm[0,
#                   0:train_num] - 1
#     remain_num = Ndata - train_num
#     val_num = remain_num
#
#     val_index = indexperm[0, train_num:train_num + val_num] - 1
#
#
#     #用全0去替代原有视图
#     # for i in range(WE.shape[0]):
#     #     for j in range(num_view):
#     #         if WE[i][j]==0:
#     #             X[j][i] = np.zeros(X[j].shape[1])
#     # WE = np.ones((num_data, num_view))
#     # incomplete label construction
#
#
#     if label.min() == -1:
#         label = (label + 1) * 0.5
#
#     mul_X_trian = [X[xiv][train_index] for xiv in range(len(X))]
#     mul_X_val = [X[xiv][val_index] for xiv in range(len(X))]
#     WE_train = get_mask(num_view, train_index.shape[0], args.miss_rate)
#     WE_val = get_mask(num_view, val_index.shape[0], args.miss_rate)
#
#     # for i in range(WE_train.shape[0]):
#     #     for j in range(num_view):
#     #         if WE_train[i][j]==0:
#     #             mul_X_trian[j][i] = np.zeros(X[j].shape[1])
#     # for i in range(WE_val.shape[0]):
#     #     for j in range(num_view):
#     #         if WE_val[i][j]==0:
#     #             mul_X_val[j][i] = np.zeros(X[j].shape[1])
#     #
#     # WE_train = np.ones((train_index.shape[0],num_view))
#     # WE_val = np.ones((val_index.shape[0],num_view))
#
#     WE_train = torch.tensor(WE_train).float()
#     WE_val = torch.tensor(WE_val).float()
#
#
#
#
#     for iv in range(view_num):
#         mul_X[iv] = np.copy(X[iv])
#         mul_X[iv] = mul_X[iv].astype(np.float32)
#         mul_X[iv] = torch.tensor(mul_X[iv])
#
#     mul_X_trian = [F.normalize(torch.tensor(xiv).float()) for xiv in mul_X_trian]
#     mul_X_val = [F.normalize(torch.tensor(xiv).float()) for xiv in mul_X_val]
#
#
#
#     word_embedding_train = F.normalize(torch.tensor(word_embedding))
#     word_embedding_test = F.normalize(torch.tensor(word_embedding))
#
#     # print(word_embedding_test)
#
#
#
#
#     args.n_input = [xiv.shape[1] for xiv in mul_X]
#     args.n_input.append(word_embedding_test.shape[1])
#     print(args.n_input)
#
#     yt_label_noisy = np.copy(label[train_index])
#     yv_label_noisy = np.copy(label[val_index])
#
#     yt_label_clean = np.copy(label_clean[train_index])
#     yv_label_clean = np.copy(label_clean[val_index])
#
#
#
#
#
#     train_max_acc, val_max_acc = train_DIC(mul_X_trian, mul_X_val, WE_train, WE_val, yt_label_noisy, yv_label_noisy, yt_label_clean,yv_label_clean, word_embedding_train,word_embedding_test,device, args)
#     return train_max_acc, val_max_acc



import time

def single_dataset_ratio(noise):
    filename = "./A_result.txt".format(dataset)
    zz = [2025,2026,2027,2028,2029]
    # zz = [2024]
    # zz = [2024]
    args.data_path = './Datasets/' + args.dataset
    args.noiseRatio = noise
    val_accs = []
    raight_rights = []
    for item in zz:
        st = time.time()
        set_seed(item)
        train_acc, val_acc = main()
        val_accs.append(val_acc)
        txt = "{}（{}） seed:{}   lr:{}   {:.4f}    {:.4f}   {:.5f}   k:{}    q:{}    tau:{}           ===>{}<===\n".format(
            args.dataset, args.miss_rate, item,args.lrkl, args.beta, args.gamma,args.alpha, args.k, args.q, args.tau, val_acc)
        f1 = open("detail.txt", encoding='utf-8', mode='a')
        f1.write(txt)
        f1.close()
        et = time.time()
        ct = (et - st) / 60
        print("花费时间：{:.2f}分钟".format(ct))
    avg = np.mean(val_accs)
    std = np.std(val_accs)
    res = "{:.2f}+{:.2f}".format(avg, std)
    txt = "{}({:.2f})   lr:{}   alpha:{}    beta:{}   gamma:{:.5f}   k:{}    q:{}    tau:{}         ===>{}<===\n".format(args.dataset,args.miss_rate,args.lrkl,args.alpha,args.beta,args.gamma,args.k,args.q,args.tau,res)
    f1 = open(filename, encoding='utf-8', mode='a')
    f1.write(txt)
    f1.close()
    print(txt)
    return "{:.2f}+{:.2f}".format(avg, std)


if __name__ == "__main__":
    ##   PIE, Caltech101, UCI, BBC, Leaves Scene LandUse_21
    # single_dataset_ratio("Caltech101",0.5, 0.5)
    ratios = np.arange(0.5, 0.55, 0.10)
    # caltech101-20  Scene15_3view_pca  My_Leaves  My_LandUse21  My_Scene15  My_Scene15_new  My_caltech101-20
    # datasets = ["CUB"]
    datasets = ["UCI","Caltech101","Leaves"]
    print("新方法")
    args.epochs = 100
    miss_rates = [0.9,0.1,0.3,0.5,0.7,0.9]
    alphas = [3e1]  #distill loss
    betas = [1e-2]   #word cls loss
    gammas = [1e0]    #contrastive loss
    value = []
    args.alpha = 0.1     #图损失no
    args.beta = 0.1 #对比学习损失
    args.gamma = 0  #相关性损失
    args.tau = 0.5    # tau
    args.q = 0.3    # q
    args.k = 1      #外部知识匹配数量
    for dataset in datasets:
        if dataset == "My_caltech101_new" or dataset == "My_La6ndUse21_new":
            args.lrkl = 0.01
        elif dataset == "CUB":
            args.lrkl = 0.01
        elif dataset == "Caltech101":
            args.lrkl = 0.01
        elif dataset == "UCI":
            args.lrkl = 0.005
        elif dataset == "LandUse21":
            args.lrkl = 0.005
        elif dataset == "PIE":
            args.lrkl = 0.01
        elif dataset == "BBC":
            args.lrkl = 0.001
        elif dataset == "cifa10":
            args.lrkl = 0.01
        elif dataset == "Leaves":
            args.lrkl = 0.005
        elif dataset == "Scene15":
            args.lrkl = 0.005
        elif dataset == "ALOI":
            args.lrkl = 0.003         #不手动加归一化时0.002最好
        elif dataset == "fminist":
            args.lrkl = 0.05
        elif dataset == "tinyimage":
            args.lrkl = 0.005
        else:
            args.lrkl = 10000
        args.dataset = dataset
        ress = []
        for alpha in alphas:
            args.alpha = alpha
            for beta in betas:
                args.beta = beta
                for gamma in gammas:
                    args.gamma = gamma
                    for miss_rate in miss_rates:
                        args.miss_rate = miss_rate
                        args.correct=True
                        res = single_dataset_ratio(0)
                        value.append(res)
                        ress.append(res)