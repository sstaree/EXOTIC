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
from model import DICNet
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
parser.add_argument('--sigma', type=float, default=30)
parser.add_argument('--tau', type=float, default=0.5)
parser.add_argument('--q', type=float, default=0.1)
parser.add_argument('--momentumkl', type=float, default=0.5)
parser.add_argument('--knn_threshold', type=float, default=0)
parser.add_argument('--high_scor', default=0.95, type=float)
parser.add_argument('--num_class', default=1.0, type=float)
parser.add_argument('--correct', default=True, type=bool)
parser.add_argument('--imputation', default=True, type=bool)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--knowledge_number', default=500, type=int)
parser.add_argument('--knowledge_base', default="Caltech101", type=str)

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

def count_params_in_m(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_m = total / 1e6
    trainable_m = trainable / 1e6
    print(f"Total parameters: {total_m:.3f} M")
    print(f"Trainable parameters: {trainable_m:.3f} M")
    return total_m, trainable_m



# loss = nn.CrossEntropyLoss()
loss = mseLoss
loss1 = My_loss()
loss2 = My_BCE_loss()


def train_DIC(mul_X, mul_X_val, WE, WE_val, yt_label_noisy, yv_label, yt_label_clean, yv_label_clean,word_embedding_train, device, args):
    # return None, torch.randn(9, 1)
    model = DICNet(
        n_stacks=4,
        n_input=args.n_input,
        n_z=args.N_z,
        Nlabel=args.Nlabel).to(device)
    
    count_params_in_m(model)

    # exit()
    
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
    optimizer = SGD(model.parameters(), lr=args.lrkl, momentum=args.momentumkl)
    # optimizer = Adam(model.parameters(), lr=args.lrkl)
    plt_val_acc = []
    plt_train_acc = []
    plt_val_loss = []
    plt_val_loss = []
    plt_train_loss = []
    st = time.time()
    train_st = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        index_array = np.arange(num_X)
        if args.AE_shuffle == True:
            np.random.shuffle(index_array)


        train_loss,train_acc_fea,train_acc_word,cont_loss,dis_loss = warmup(epoch,mul_X_1,yt_label_clean,yt_label_correct,word_embedding_train,args,model,device,optimizer,loss_model,loss,WE,index_array)
        if (epoch+1)%5==0 or epoch == 0:
            tst = time.time()
            val_loss, val_acc_fea,val_acc_word,fusion_z_val,_ = test(epoch, mul_X_val, yv_label_clean, yv_label, args, model, device, optimizer, loss_model,loss, WE_val, index_array, word_embedding_train)
            tet = time.time()
            print("推理花的时间：",(tet-tst)/60/60)
            plt_val_loss.append(val_loss)
            plt_val_acc.append(val_acc_fea)
            plt_train_acc.append(train_acc_fea)
            plt_train_loss.append(train_loss)
            print(
                "epoch:{}    train_loss:{:.4f}    val_loss:{:.4f}   train_acc:{:.4f}       cont_loss:{:.2f}    dis_loss:{:.2f}     val_acc_fea:{:.4f}  val_acc_word:{:.4f}".format(epoch+1, train_loss,val_loss,
                                                                                             train_acc_fea,cont_loss,dis_loss,
                                                                                             val_acc_fea,val_acc_word))
    # trian_et = time.time()
    # print("训练花的时间：",(trian_et-train_st)/60/60)
    # tst = time.time()
    # val_loss, val_acc_fea,val_acc_word,fusion_z_val,_ = test(epoch, mul_X_val, yv_label_clean, yv_label, args, model, device, optimizer, loss_model,loss, WE_val, index_array, word_embedding_train)
    # tet = time.time()
    # print("推理花的时间：",(tet-tst)/60/60)
    # et = time.time()
    # ct = (et-st)/60/60
    # print("总的花费时间：{:.2f}h".format(ct))
    # exit()
    # torch.save(model, "D:\python project\MVMLC\新方法 - 副本\{}({})_model.pth".format(args.dataset,args.miss_rate))
    # with open(r'D:\python project\MVMLC\EXOTIC\tsne_feature_label\feature_{}_{}.pkl'.format(args.dataset,args.miss_rate), 'wb') as file:
    #     pickle.dump(fusion_z_val.cpu().numpy(), file)
    # with open(r'D:\python project\MVMLC\EXOTIC\tsne_feature_label\label_{}_{}.pkl'.format(args.dataset, args.miss_rate), 'wb') as file:
    #     pickle.dump(np.argmax(yv_label_clean,axis=1), file)
    # exit()

    # mytsne(fusion_z_val.cpu().numpy(),np.argmax(yv_label_clean,axis=1),"Tsne_{}_0.9".format(args.dataset))
    # exit()

    return plt_train_acc[-1] * 100, plt_val_acc[-1] * 100



import scipy.io as sio

def main():
    global args
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    X, Y_clean , word_embedding = import_and_load_data(args.dataset,args.knowledge_base)

    word_embedding = word_embedding[0:args.knowledge_number,:]

    X_train, X_test, y_train, y_test = balanced_train_test_split(X, Y_clean)

    y_train = np.hstack((y_train, np.zeros((y_train.shape[0], 4))))
    y_test = np.hstack((y_test, np.zeros((y_test.shape[0], 4))))

    X_train_tensor = [torch.tensor(ivx).float() for ivx in X_train]
    X_test_tensor = [torch.tensor(ivx).float() for ivx in X_test]

    num_view = len(X)
    WE_train = get_mask(num_view, X_train[0].shape[0], args.miss_rate)
    WE_val = get_mask(num_view, X_test[0].shape[0], args.miss_rate)

    # print(WE_train)
        

    WE_train = torch.tensor(WE_train).float()
    WE_val = torch.tensor(WE_val).float()

    # word_embedding = np.random.randn(50000, 768)

    word_embedding_train = F.normalize(torch.tensor(word_embedding)).float()
    # word_embedding_train = torch.tensor(word_embedding) 

    # word_embedding_train = F.normalize(torch.randn((50000,768)))
    # word_embedding_test = F.normalize(torch.randn((50000,768)))

    args.Nlabel = y_train.shape[1]
    args.N_z = word_embedding_train.shape[1]
    # args.N_z = args.Nlabel
    args.n_input = [xiv.shape[1] for xiv in X_train]
    args.n_input.append(word_embedding_train.shape[1])
    print(args.n_input)

    print(type(y_train))

    # print(WE_val)
    # print(np.argmax(y_test,axis=1))
    im = []    # 各类别instance个数

    ie = []     # 各类别instance嵌入

    for i in range(args.Nlabel):
        im.append([0]*len(X_train))
        ie.append([[] for i in range(len(X_train))])

    print(im)
    print(ie)
    for i in range(X_train[0].shape[0]):
        for v in range(len(X_train)):
            if(WE_train[i,v].item() > 0):
               im[int(np.argmax(y_train[i]))][v] +=1
               ie[int(np.argmax(y_train[i]))][v].append(X_train[v][i])
    print("训练集大小：",y_train.shape[0],"测试集大小",y_test.shape[0])

    train_max_acc, val_max_acc = train_DIC(X_train_tensor, X_test_tensor, WE_train, WE_val, y_train, y_test, y_train,y_test, word_embedding_train,device, args)
    return train_max_acc, val_max_acc




import time

def single_dataset_ratio(noise):
    filename = r"D:\所有结果重新复核\InfoNCE.txt".format(dataset)
    zz = [2025,2026,2027,2028,2029]
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
        f1 = open("D:\python project\MVMLC\EXOTIC\A_detail.txt", encoding='utf-8', mode='a')
        f1.write(txt)
        f1.close()
        et = time.time()
        ct = (et - st) / 60
        print("花费时间：{:.2f}分钟".format(ct))
    avg = np.mean(val_accs)
    std = np.std(val_accs)
    res = "{:.2f}+{:.2f}".format(avg, std)
    txt = "{}({:.2f})   lr:{}   alpha:{}    beta:{:.5f}   gamma:{:.5f}   k:{}    q:{}    tau:{}    knowledge_base:{}    knowledge_number:{:>5d}         ===>{}<===\n".format(args.dataset,args.miss_rate,args.lrkl,args.alpha,args.beta,0,args.k,args.q,args.tau,args.knowledge_base,args.knowledge_number,res)
    f1 = open(filename, encoding='utf-8', mode='a')
    f1.write(txt)
    f1.close()
    f2 = open("D:\python project\MVMLC\EXOTIC\A_detail.txt", encoding='utf-8', mode='a')
    f2.write(txt)
    f2.close()
    print(txt)
    return "{:.2f}+{:.2f}".format(avg, std)

if __name__ == "__main__":
    datasets = ["HW","Scene15","LandUse21","Caltech101","Fashion","CUB","NUSWIDEOBJ"]
    datasets = ["HW","Caltech101"]
    knowledege_bases = ["ImageNet"]
    print("新方法")
    args.epochs = 100     
    miss_rates = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    miss_rates = [0.9] 
    # args.knowledge_base   = "ImageNet"
    value = [] 

    args.tau = 0.5    # tau.
    args.sigma = 0  

    args.q = 0.3       # q
    args.k = 100       #外部知识匹配数量

    args.gamma = 0
    


    alphas = [30]
    betas = [1]
    qs =  [0.3]
    # knowledge_numbers = [100,1000,5000,10000,50000]
    knowledge_numbers = [100,1000,5000,10000,50000]
    for knowledge_base in knowledege_bases:
        args.knowledge_base = knowledge_base
        for dataset in datasets:
            if dataset == "My_caltech101_new" or dataset == "My_La6ndUse21_new":
                args.lrkl = 0.01
            elif dataset == "CUB":
                args.lrkl = 0.01
            elif dataset == "Caltech101":
                args.lrkl = 0.01
                # args.beta = 1
            elif dataset == "UCI":
                args.lrkl = 0.005
            elif dataset == "LandUse21":
                args.lrkl = 0.005
            elif dataset == "PIE":
                args.lrkl = 0.01
            elif dataset == "NUSWIDEOBJ":
                args.lrkl = 0.001
            elif dataset == "BBC":
                args.lrkl = 0.001
            elif dataset == "HW":
                args.lrkl = 0.01
                # args.lrkl = 0.1
            elif dataset == "fashion" or dataset == "Fashion":
                args.lrkl = 0.01
                # args.lrkl = 0.5
            elif dataset == "cifa10":
                args.lrkl = 0.01
                args.epochs = 100
            elif dataset == "Leaves":
                args.lrkl = 0.05
            elif dataset == "Youtube":
                args.lrkl = 0.01
            elif dataset == "Scene15":
                args.lrkl = 0.001
            elif dataset == "ALOI":
                args.lrkl = 0.003         #不手动加归一化时0.002最好
            elif dataset == "fminist":
                args.lrkl = 0.05
            elif dataset == "tinyimage":
                args.lrkl = 0.005
            elif dataset == "MNIST":
                args.lrkl = 0.001
            else:
                args.lrkl = 10000

            args.dataset = dataset
            ress = []
            for alpha in alphas:
                args.alpha = alpha
                for beta in betas:
                    args.beta = beta
                    for q in qs:
                        args.q = q
                        for miss_rate in miss_rates:
                            args.miss_rate = miss_rate
                            for knowledge_number in knowledge_numbers:
                                args.knowledge_number = knowledge_number
                              
                                args.correct=True
                                res = single_dataset_ratio(0)
                                value.append(res)
                                ress.append(res)    
