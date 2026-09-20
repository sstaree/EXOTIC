
# this code is released by: 
# L. Yang, X.-Z. Wu, Y. Jiang, and Z.-H. Zhou. 
# Multi-label deep forest. 
#In: Proceedings of the 24th European Conference on Artificial Intelligence (ECAI'20), 
# Santiago de Compostela, Spain, 2020. [code]

import numpy as np
from sklearn import metrics
import random
import torch
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD,AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import scipy.io
import h5py
import math
from loss import Loss
import pickle
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans

def do_metric(y_prob, label):
    y_predict = y_prob > 0.5
    ranking_loss = 1 - compute_ranking_loss(y_prob, label)
    # print(ranking_loss)
    one_error = compute_one_error(y_prob, label)
    # print(one_error)
    coverage = compute_coverage(y_prob, label)
    # print(coverage)
    hamming_loss = 1 - compute_hamming_loss(y_predict, label)
    # print(hamming_loss)
    precision = compute_average_precision(y_prob, label)
    # print(precision)
    macro_f1 = compute_macro_f1(y_predict, label)
    # print(macro_f1)
    micro_f1 = compute_micro_f1(y_predict, label)
    # print(micro_f1)
    auc = compute_auc(y_prob, label)
    auc_me = mlc_auc(y_prob, label)
    return np.array([hamming_loss, one_error, coverage, ranking_loss, precision, auc, auc_me, macro_f1, micro_f1])


def init_supervise(supervise):
    if supervise == "ranking loss":
        ranking_loss = 1.0
        return ranking_loss
    elif supervise == "hamming loss":
        hamming_loss = 1.0
        return hamming_loss
    elif supervise == "one-error":
        one_error = 1.0
        return one_error
    elif supervise == "average precision":
        average_precision = 0.0
        return average_precision
    elif supervise == "micro-f1":
        micro_f1 = 0.0
        return micro_f1
    elif supervise == "macro-f1":
        macro_f1 = 0.0
        return macro_f1
    elif supervise == "coverage":
        coverage = 1000.0
        return coverage
    elif supervise == "macro_auc":
        macro_auc = 0.0
        return macro_auc


def compare_supervise_value(supervise, supervise_value1, supervise_value2):
    if supervise == "ranking loss" or supervise == "hamming loss" or supervise == "one-error" or supervise == "coverage":
        if supervise_value1 > supervise_value2 + 1e-4:
            return False
        else:
            return True
    elif supervise == "average precision" or supervise == "micro-f1" or supervise == "macro-f1" or supervise == "macro_auc":
        if supervise_value1 + 1e-4 < supervise_value2:
            return False
        else:
            return True


def compute_supervise(supervise, y_prob, label, threshold):
    predict = y_prob > threshold
    if supervise == "ranking loss":
        temp_ranking_loss = compute_ranking_loss(
            y_prob, label)  # prob / y_prob
        value = temp_ranking_loss
    elif supervise == "hamming loss":
        temp_hamming_loss = compute_hamming_loss(predict, label)
        value = temp_hamming_loss
    elif supervise == "one-error":
        temp_one_error = compute_one_error(y_prob, label)
        value = temp_one_error
    elif supervise == "average precision":
        temp_average_precision = compute_average_precision(y_prob, label)
        value = temp_average_precision
    elif supervise == "micro-f1":
        temp_micro_f1 = compute_micro_f1(predict, label)
        value = temp_micro_f1
    elif supervise == "macro-f1":
        temp_macro_f1 = compute_macro_f1(predict, label)
        value = temp_macro_f1
    elif supervise == "coverage":
        temp_coverage = compute_coverage(y_prob, label)
        value = temp_coverage
    elif supervise == "macro_auc":
        macro_auc = compute_auc(y_prob, label)
        value = macro_auc
    return value



def compute_supervise_vec(supervise, y_prob, label, threshold):
    predict = y_prob > threshold
    if supervise == "ranking loss":
        temp_ranking_loss = compute_ranking_loss_vec(
            y_prob, label)  # prob / y_prob
        value = temp_ranking_loss
    elif supervise == "hamming loss":
        temp_hamming_loss = compute_hamming_loss_vec(predict, label)
        value = temp_hamming_loss
    elif supervise == "one-error":
        temp_one_error = compute_one_error_vec(y_prob, label)
        value = temp_one_error
    elif supervise == "average precision":
        temp_average_precision = compute_average_precision_vec(y_prob, label)
        value = temp_average_precision
    elif supervise == "coverage":
        temp_coverage = compute_coverage_vec(y_prob, label)
        value = temp_coverage
    elif supervise == "macro_auc":
        macro_auc = compute_auc_vec(y_prob, label)
        value = macro_auc
    return value


def update_supervise(supervise, value_pool, layer_index, y_prob, label, threshold):
    back = False
    back2 = False
    value_pool[layer_index] = compute_supervise(
        supervise, y_prob, label, threshold)
    if layer_index >= 2 and compare_supervise_value(supervise, value_pool[layer_index - 2],
                                                    value_pool[layer_index - 1]):
        back2 = True
    if layer_index >= 1 and compare_supervise_value(supervise, value_pool[layer_index - 1], value_pool[layer_index]):
        back = True
    return [back, back2]


def compute_accuracy(pred_label, label):
    num_samples = len(label)
    acc = sum(label == pred_label) * 1.0 / num_samples
    return acc


def compute_performance_single_label(predict_score, label):
    predict_label = predict_score > 0.5
    _, num_labels = label.shape
    acc = np.empty(num_labels)
    f1 = np.empty(num_labels)
    auc = np.empty(num_labels)
    for i in range(num_labels):
        acc[i] = metrics.accuracy_score(
            label[:, i].reshape(-1), predict_label[:, i].reshape(-1))
        f1[i] = metrics.f1_score(
            label[:, i].reshape(-1), predict_label[:, i].reshape(-1))
        auc[i] = metrics.roc_auc_score(
            label[:, i].reshape(-1), predict_score[:, i].reshape(-1))
    return [acc, f1, auc]


def compute_rank(y_prob):
    rank = np.zeros(y_prob.shape)
    for i in range(len(y_prob)):
        temp = y_prob[i, :].argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(y_prob[i, :]))
        rank[i, :] = ranks
    return y_prob.shape[1] - rank


# example based measure
def compute_hamming_loss(pred_label, label):
    acc = compute_accuracy(pred_label, label)
    return 1 - acc.mean()


def compute_hamming_loss_vec(pred_label, label):
    acc = compute_accuracy(pred_label, label)
    return 1 - acc


# label based measure
def compute_macro_f1(pred_label, label):
    up = np.sum(pred_label * label, axis=0)
    down = np.sum(pred_label, axis=0) + np.sum(label, axis=0)
    if np.sum(np.sum(label, axis=0) == 0) > 0:
        up[down == 0] = 0
        down[down == 0] = 1
    macro_f1 = 2.0 * np.sum(up / down)
    macro_f1 = macro_f1 * 1.0 / label.shape[1]
    return macro_f1


def compute_micro_f1(pred_label, label):
    up = np.sum(pred_label * label)
    down = np.sum(pred_label) + np.sum(label)
    if np.sum(np.sum(label) == 0) > 0:
        up[down == 0] = 0
        down[down == 0] = 1
    micro_f1 = 2.0 * up / down
    return micro_f1


# ranking based measure
def compute_ranking_loss(y_prob, label):
    # y_predict = y_prob > 0.5
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        prob_positive = y_prob[i, label[i, :] > 0.5]
        prob_negative = y_prob[i, label[i, :] < 0.5]
        s = 0
        for j in range(prob_positive.shape[0]):
            for k in range(prob_negative.shape[0]):
                if prob_negative[k] >= prob_positive[j]:
                    s += 1

        label_positive = np.sum(label[i, :] > 0.5)
        label_negative = np.sum(label[i, :] < 0.5)
        if label_negative != 0 and label_positive != 0:
            loss = loss + s * 1.0 / (label_negative * label_positive)

    return loss * 1.0 / num_samples


def compute_ranking_loss_vec(y_prob, label):
    num_samples, num_labels = label.shape
    loss = np.zeros(num_samples)
    for i in range(num_samples):
        prob_positive = y_prob[i, label[i, :] > 0.5]
        prob_negative = y_prob[i, label[i, :] < 0.5]
        s = 0
        for j in range(prob_positive.shape[0]):
            for k in range(prob_negative.shape[0]):
                if prob_negative[k] >= prob_positive[j]:
                    s += 1

        label_positive = np.sum(label[i, :] > 0.5)
        label_negative = np.sum(label[i, :] < 0.5)
        if label_negative != 0 and label_positive != 0:
            loss[i] = s * 1.0 / (label_negative * label_positive)
    return loss


def compute_one_error(y_prob, label):
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        pos = np.argmax(y_prob[i, :])
        loss += label[i, pos] < 0.5
    return loss * 1.0 / num_samples


def compute_one_error_vec(y_prob, label):
    num_samples, num_labels = label.shape
    loss = np.zeros(num_samples)
    for i in range(num_samples):
        pos = np.argmax(y_prob[i, :])
        loss[i] = label[i, pos] < 0.5
    return loss


def compute_coverage(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    coverage = 0
    for i in range(num_samples):
        if sum(label[i, :] > 0.5) > 0:
            coverage += max(rank[i, label[i, :] > 0.5])
    coverage = coverage * 1.0 / num_samples - 1
    return coverage / num_labels


def compute_coverage_vec(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    coverage = np.zeros(num_samples)
    for i in range(num_samples):
        if sum(label[i, :] > 0.5) > 0:
            coverage[i] = max(rank[i, label[i, :] > 0.5])
    return coverage


def compute_average_precision(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    precision = 0
    for i in range(num_samples):
        positive = np.sum(label[i, :] > 0.5)
        rank_i = rank[i, label[i, :] > 0.5]
        temp = rank_i.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(rank_i))
        ranks = ranks + 1
        ans = ranks * 1.0 / rank_i
        if positive > 0:
            precision += np.sum(ans) * 1.0 / positive
    return precision / num_samples


def compute_average_precision_vec(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    precision = np.zeros(num_samples)
    for i in range(num_samples):
        positive = np.sum(label[i, :] > 0.5)
        rank_i = rank[i, label[i, :] > 0.5]
        temp = rank_i.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(rank_i))
        ranks = ranks + 1
        ans = ranks * 1.0 / rank_i
        if positive > 0:
            precision[i] = np.sum(ans) * 1.0 / positive
    return precision


def compute_auc(y_prob, label):
    n, m = label.shape
    macro_auc = 0
    valid_labels = 0
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            index = np.argsort(y_prob[:, i])
            pred = y_prob[:, i][index]
            y = label[:, i][index] + 1
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
            temp = metrics.auc(fpr, tpr)
            macro_auc += temp
            valid_labels += 1
    macro_auc /= valid_labels
    return macro_auc

def compute_mlr_auc(y_prob, label):
    n, m = label.shape
    macro_auc = 0
    valid_labels = 0
    fpr = np.zeros(m)
    tpr = np.zeros(m)
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            index = np.argsort(y_prob[:, i])
            pred = y_prob[:, i][index]
            y = label[:, i][index] + 1
            fpr[i], tpr[i], thresholds = metrics.roc_curve(y, pred, pos_label=2)
    area = 0
    for i in range(m):
        area = area+(fpr[i+1]-fpr[i])*(tpr[i+1]+tpr[i])*0.5
    mlr_auc = area/(fpr[m]-fpr[1])
    return mlr_auc

def compute_auc_vec(y_prob, label):
    n, m = label.shape
    macro_auc = np.zeros(m)
    valid_labels = 0
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            index = np.argsort(y_prob[:, i])
            pred = y_prob[:, i][index]
            y = label[:, i][index] + 1
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
            temp = metrics.auc(fpr, tpr)
            macro_auc[i] = temp
            valid_labels += 1
    return macro_auc

def performance(y,f,T):
#    code is written by Jerry, according to the original code from 
#    from http://mlda.swu.edu.cn/codes.php?name=iMVWL
    n,K = f.shape    
    match = np.zeros(n)
    fn = np.zeros(n)
    fp = np.zeros(n)
    for i in range(n):
        si = f[i,:].argsort()[::-1]        
        words=y[i,:]
        correct_labels=np.where(words>-1)
        correct_labels = (np.array(correct_labels)).reshape(-1)
        si = si[0:T]   # T numbers
        match[i] = 0
        for j in range(len(correct_labels)):
            if np.where(si==correct_labels[j])[0].shape[0]!=0:
                match[i] = match[i]+1
        fn[i] = len(correct_labels)-match[i]
        fp[i] = T-match[i]
    return match,fp,fn
  
def mlr_roc(f, y_test):
#    code is written by Jerry, according to the original code from 
#    from http://mlda.swu.edu.cn/codes.php?name=iMVWL
    K = y_test.shape[1]
    tpr1 = np.zeros(K)
    fpr1 = np.zeros(K)
    
    for i in range(K):
        match,fpp,fnn = performance(y_test,f,i+1);
        tp1=match.sum()
        fn1=fnn.sum()
        fp1=fpp.sum()
        tn1 = K*f.shape[0]-(tp1+fp1+fn1)
        tpr1[i] = tp1/(tp1+fn1)
        fpr1[i] = fp1/(fp1+tn1)
    return tpr1,fpr1  
def mlc_auc(rocZ,newY):
#    code is written by Jerry, according to the original code from 
#    from http://mlda.swu.edu.cn/codes.php?name=iMVWL    
#    rocZ: problistic matrix  n*c
#    newY: n*c matrix,elements in {-1,1}
    if newY.min()==0:
        newY = newY*2-1
    
    tpr,fpr = mlr_roc(rocZ,newY)
    area = 0
    m = newY.shape[1]
    for i in range(m-1):
        area = area+(fpr[i+1]-fpr[i])*(tpr[i+1]+tpr[i])*0.5
    value_auc = area/(fpr[m-1]-fpr[0])
    return value_auc

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
    regularization_loss = 0                    # 正则项
    for param in model.parameters():
        regularization_loss += torch.sum(param ** 2)                  # 计算所有参数平方
    return loss(pred, target) + 0.001 * regularization_loss             # 返回损失。

import scipy.io as sio
def load_matmultiview_data(mat_file_path):
    # 加载mat文件
    mat_data = sio.loadmat(mat_file_path)

    # 找到所有视图的键名 'views0', 'views1', ..., 'viewsv-1'
    views = []
    v = 0  # 初始化视图数量
    while True:
        view_key = f'views{v}'
        if view_key in mat_data:
            views.append(mat_data[view_key])  # 将视图数据添加到列表中
            v += 1
        else:
            break  # 一旦找不到下一个视图的键名，结束循环
    y = mat_data['labels']
    embeddings = mat_data['embeddings']
    # 返回视图数据列表和视图数量
    return views, v ,y ,embeddings


def normalize_features(matrix):
    """
    对每个特征维度进行归一化 (n * c)，使每个特征维度缩放到 [0, 1] 范围。
    :param matrix: 输入的 n * c 矩阵，n 是样本数，c 是特征维度。
    :return: 归一化后的矩阵
    """
    # 找到每列的最小值和最大值
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)

    # 防止分母为0的情况
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # 如果某一列中所有值相等，则直接将范围设为1，防止除以0

    # 归一化操作
    normalized_matrix = (matrix - min_vals) / ranges

    return normalized_matrix


from sklearn.model_selection import train_test_split


def balanced_train_test_split(features_dict, labels, test_size=0.2, random_state=None):
    """
    平衡划分多视图数据的训练集和测试集，确保每个类别在训练集和测试集中的比例一致。

    参数：
    - features_dict: 字典，包含每个视图的特征数据，键是视图名，值是该视图的特征矩阵。
    - labels: numpy矩阵，独热编码的标签。
    - test_size: 测试集比例，默认0.2。
    - random_state: 随机种子，保证结果可重现。

    返回：
    - X_train: 训练集特征（列表形式）。
    - X_test: 测试集特征（列表形式）。
    - y_train: 训练集标签。
    - y_test: 测试集标签。
    """
    # 获取标签的类别数
    num_classes = labels.shape[1]

    # 获取每个类别的索引
    class_indices = [np.where(labels[:, i] == 1)[0] for i in range(num_classes)]

    # 分别从每个类别中进行均衡抽样
    train_indices = []
    test_indices = []

    for indices in class_indices:
        # 对每个类别中的样本索引进行划分
        train, test = train_test_split(indices, test_size=test_size, random_state=random_state)
        train_indices.extend(train)
        test_indices.extend(test)

    # 打乱划分后的训练集和测试集
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # 按照划分的索引选择标签
    y_train, y_test = labels[train_indices], labels[test_indices]

    # 处理每个视图的特征，返回训练集和测试集
    X_train = [features_dict[view][train_indices] for view in range(len(features_dict))]
    X_test = [features_dict[view][test_indices] for view in range(len(features_dict))]

    return X_train, X_test, y_train, y_test


def compute_mutual_information(X, Y):
    # X: 输入特征，Y: 预测的概率矩阵
    N, C = Y.shape  # N是样本数量，C是类别数量

    # 计算边缘概率 P(x_i) 和 P(y_j)
    p_x = torch.sum(Y, dim=0) / N  # 计算每个类别的边缘概率 P(y_j)
    p_y = torch.sum(X, dim=0) / N  # 计算每个特征的边缘概率 P(x_i)

    # 计算联合概率 P(x_i, y_j)
    joint_prob = torch.matmul(X.T, Y) / N

    # 计算互信息
    mutual_info = torch.sum(joint_prob * torch.log(joint_prob / (p_x.unsqueeze(1) * p_y.unsqueeze(0))))

    return mutual_info





def balanced_kmeans(X, c, max_iter=100, lambda_penalty=0.1):
    """
    X: 输入特征矩阵，形状 (N, F)，N 是样本数量，F 是特征维度
    c: 类别数量
    max_iter: 最大迭代次数
    lambda_penalty: 类别平衡惩罚项的权重
    """
    N, F = X.shape
    kmeans = KMeans(n_clusters=c, max_iter=max_iter)

    # 初步K-means聚类
    kmeans.fit(X.detach().cpu().numpy())
    labels = torch.tensor(kmeans.labels_)

    # 计算类别样本数量
    class_counts = torch.bincount(labels)
    target_count = N // c  # 理想的样本数量
    penalty = torch.sum(torch.abs(class_counts - target_count))

    # 计算加权的损失，平衡类间样本数量
    # 在此例子中，使用了一个简单的平衡惩罚项
    loss = 0
    for i in range(c):
        class_data = X[labels == i]
        center = torch.mean(class_data, dim=0)
        loss += torch.sum((class_data - center) ** 2)

    # 加入类别平衡的惩罚项
    loss += lambda_penalty * penalty
    return labels, loss
from sklearn.manifold import TSNE
def mytsne(h,y ,filename):
    X = h
    perplexity = 20
    tsne = TSNE(n_components=2, random_state=33, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    num_label = np.max(y)-np.min(y)+1
    plt.figure(figsize=(8, 6))
    if np.min(y) == 1:
        for i in range(1, num_label + 1):
            plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=str(i))
    else:
        for i in range(num_label):
            plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=str(i))

    plt.xticks([])
    plt.yticks([])
    plt.savefig('{}.pdf'.format(filename), dpi=300, format='pdf')
    plt.show()
    plt.close()


from sklearn.mixture import GaussianMixture


def gmm_matching(z_word, z1, n_components):
    """
    通过高斯混合模型（GMM）对外部知识库的表示进行建模，为输入特征矩阵中的每个样本匹配一个外部知识。

    参数：
    - z_word: 外部知识表示矩阵，PyTorch Tensor，形状为 (n_word, c)
    - z1: 输入特征矩阵，PyTorch Tensor，形状为 (n1, c)
    - n_components: GMM 中的高斯组件数，即类别数

    返回：
    - labels: 输入特征矩阵 z1 中每个样本对应的外部知识的类别标签，返回 PyTorch Tensor
    """
    # 将 Tensor 转换为 NumPy 数组
    z_word_np = z_word.cpu().numpy()  # 转换为 NumPy 数组
    z1_np = z1.cpu().numpy()  # 转换为 NumPy 数组

    # 使用高斯混合模型（GMM）对外部知识表示进行建模
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(z_word_np)  # 通过外部知识表示矩阵拟合 GMM

    # 通过 GMM 模型预测输入特征矩阵 z1 的每个样本最匹配的类别
    labels = gmm.predict(z1_np)  # 返回一个类别标签数组

    # 将 NumPy 数组转换回 PyTorch Tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(z1.device)

    return labels_tensor

def import_and_load_data(mu,knowledge_base):
    # load data
    with open(r"D:\python project\MVMLC\Dataset create\自制数据集\{}\X.pkl".format(mu), 'rb') as file:
        print(file)
        X = pickle.load(file)
    with open(r"D:\python project\MVMLC\Dataset create\自制数据集\{}\Y.pkl".format(mu) , 'rb') as file:
        Y_true = pickle.load(file)
    #D:\python project\MVMLC\Dataset create\Imagenet_Embedding_5W.pkl       D:\浏览器下载\2024-ICML-TAC-main\2024-ICML-TAC-main\data\nouns_embedding_ensemble.pkl
    #"D:\python project\MVMLC\Dataset create\自制数据集\Caltech101\Embeddings_Caltech101.pkl"
    #"D:\python project\MVMLC\Dataset create\自制数据集\Leaves\Embeddings.pkl"
    #"D:\python project\MVMLC\Dataset create\自制数据集\Scene15\Embeddings_Scene15.pkl"
    #D:\python project\MVMLC\Dataset create\personai_icartoonface_detval.pkl
    with open(r"D:\python project\MVMLC\EXOTIC\Knowledge Base\{}.pkl".format(knowledge_base) , 'rb') as file:
        Embeddings = pickle.load(file)
    Embeddings = np.squeeze(Embeddings)
    return X, Y_true , Embeddings


def compute_graph_loss(A, B, similarity='cosine'):
    """
    计算两个表示矩阵的图损失，使其空间结构尽可能一致。

    参数：
        A: torch.Tensor, 形状为 (N, c)，表示矩阵 A。
        B: torch.Tensor, 形状为 (N, c)，表示矩阵 B。
        similarity: str, 相似度度量方式，可选 'cosine' 或 'euclidean'。

    返回：
        graph_loss: torch.Tensor, 图损失标量。
    """
    # 归一化表示矩阵（适用于余弦相似度）
    A = F.normalize(A, dim=1)
    B = F.normalize(B, dim=1)

    # 计算相似度矩阵
    if similarity == 'cosine':
        S_A = torch.matmul(A, A.T)  # A 的相似度矩阵
        S_B = torch.matmul(B, B.T)  # B 的相似度矩阵
    elif similarity == 'euclidean':
        S_A = torch.cdist(A, A, p=2)  # A 的距离矩阵
        S_B = torch.cdist(B, B, p=2)  # B 的距离矩阵
        S_A = torch.exp(-S_A)  # 将距离转化为相似度（可选）
        S_B = torch.exp(-S_B)
    else:
        raise ValueError("Invalid similarity metric. Choose 'cosine' or 'euclidean'.")

    # 计算图损失（Frobenius 范数的平方）
    graph_loss = torch.norm(S_A - S_B, p='fro') ** 2
    return graph_loss
