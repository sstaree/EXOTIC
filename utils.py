import sys
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import normalize
import math
import glob, os, shutil
from loss import Loss,DistillLoss,graph_loss
import csv
from loss  import  GCELoss

from measure import *






def contrast_loss1(h0, h1,we,q=0.3,tau=0.5):
    # Step 1: 根据 we1 和 we2 创建掩码，过滤掉缺失的样本
    mask_miss_inst = we.bool()
    h1 = h1[mask_miss_inst]
    h0 = h0[mask_miss_inst]
    # Step 4: 归一化有效样本特征
    v1 = F.normalize(h0, p=2, dim=1)  # 对向量 v1 做 L2 归一化
    v2 = F.normalize(h1, p=2, dim=1)  # 对向量 v2 做 L2 归一化
    # Step 5: 计算相似度矩阵
    similarity_mat = torch.matmul(v1, v2.T) / 0.5  # 计算 v1 和 v2 之间的相似度矩阵并除以温度系数 t
    pos = torch.exp(similarity_mat/tau).diag()  # 提取对角线，表示正样本对的相似度
    fm = torch.exp(similarity_mat/tau).sum(1)  # 对每个样本的相似度总和
    P = pos / fm   #原
    # P = pos.sum() / fm.sum()
    # P = -torch.log(pos / fm)
    loss = ((1-torch.pow(P,q))/math.log(1+q)).mean()
    # loss = -torch.log(pos / fm).mean()

    return loss






def dis1_cont1(features, labels,we,q):
    mask_miss_inst = we.bool()
    features = features[mask_miss_inst]
    labels = labels[mask_miss_inst]
    sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)


    label_sim_matrix = torch.matmul(labels, labels.T) - torch.eye(features.shape[0]).cuda()

    pos = torch.exp(label_sim_matrix * sim_matrix/0.5).sum()

    fm = torch.exp((sim_matrix * (1 - torch.eye(features.shape[0]).cuda()))/0.5).sum()

    P = pos / fm
    # P = -torch.log(pos / fm)
    # loss = (((1 - torch.pow(P, q)) / math.log(1 + q))).mean()
    loss = -(torch.log(P)).mean()

    return loss



def Ldis1(features, labels,we,q, margin=1.0, lambda_dis=0):
    """
    计算判别性损失 (Ldis)，即Contrastive Loss + 类别一致性（Align）损失

    参数:
        features (torch.Tensor): 特征张量，形状为 (batch_size, feature_dim)
        labels (torch.Tensor): 真实标签的 one-hot 编码，形状为 (batch_size, num_classes)
        margin (float): 判别性损失中的边际阈值
        lambda_dis (float): 控制判别性损失与对齐损失的平衡

    返回:
        torch.Tensor: 总的判别性损失
    """
    total_loss = dis1_cont1(features,labels,we,q)



    return total_loss






def warmup(epoch,mul_X,yt_label_clean,yt_label_correct,word_embedding_train,args,model,device,optimizer,loss_model,loss,WE,index_array):
    num_X = mul_X[0].shape[0]
    train_loss = 0
    cont_loss = 0
    dis_loss = 0
    right_num =0
    right_num1 =0
    sum = 0
    matrix = WE.numpy()
    total_elements = matrix.size
    missing_elements = np.sum(matrix == 0)
    missing_rate = missing_elements / total_elements
    total_prob = 0
    for batch_idx in range(int(np.ceil(num_X / args.batch_size))):
        idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X)]
        mul_X_batch = []
        for iv, X in enumerate(mul_X):
            mul_X_batch.append(X[idx].to(device))
        we = WE[idx].to(device)
        yt_label_correct = torch.tensor(yt_label_correct)
        sub_target = yt_label_correct[idx].float().to(device)  # batch_y
        true_target = torch.tensor(yt_label_clean)[idx].float().to(device)

        optimizer.zero_grad()
        word_embedding_train = word_embedding_train.to(device)

        x_bar_list, yLable_logit,yLable_word_logit, fusion_z, individual_zs, z_word ,individual_zs_p,z_word_list,z_d ,z_word_select_all = model(mul_X_batch,we,word_embedding_train,args.k,epoch,'train')

        loss_Cont = 0
        for i in range(len(individual_zs)):
            loss_Cont += contrast_loss1(z_word, individual_zs[i],we[:,i],args.q,args.tau)
        
        loss_Cont1 = 0
        for i in range(len(individual_zs)):
            loss_Cont1 += contrast_loss1(z_word, individual_zs_p[i],we[:,i],args.q,args.tau)
        
        y_prob = torch.softmax(yLable_logit,dim=1).max(1)[0].sum()

        total_prob += y_prob.item()

        loss_CL = -torch.mean(torch.sum(F.log_softmax(yLable_logit, dim=1) * sub_target, dim=1))

        # loss_CL1 = -torch.mean(torch.sum(F.log_softmax(yLable_word_logit, dim=1) * sub_target, dim=1))

        loss_dis1 = Ldis1(fusion_z, sub_target, torch.ones((we.shape[0],)),args.q)    # EXOTIC 用

        loss_dis2 = Ldis1(z_word, sub_target, torch.ones((we.shape[0],)),args.q)
        loss_dis3 = Ldis1(z_d, sub_target, torch.ones((we.shape[0],)),args.q)

        loss_graph = graph_loss(z_d,true_target)

        loss_dis = 0.5 * (loss_dis1 + loss_dis2)

        right_num += np.sum(np.argmax(yLable_logit.detach().cpu().numpy(),axis=1)==np.argmax(true_target.detach().cpu().numpy(),axis=1))

        if epoch <0:
            # loss_fusion = args.alpha * loss_dis1  + args.alpha * loss_dis
            loss_fusion = args.alpha * loss_dis  + args.gamma * loss_Cont
        else:
            loss_fusion = loss_CL + args.alpha * loss_dis1 + args.beta * loss_Cont

        loss_fusion.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss_CL.cpu().item()
        cont_loss += loss_Cont.cpu().item()
    # if (epoch+1)%20==0:
    #     print(total_prob/num_X)
    train_loss = train_loss/num_X
    acc_fea = right_num/num_X
    acc_word = right_num1/num_X

    return train_loss,acc_fea,acc_word,cont_loss,dis_loss


# def warmup(epoch,mul_X,yt_label_clean,yt_label_correct,word_embedding_train,args,model,device,optimizer,loss_model,loss,WE,index_array):
#     num_X = mul_X[0].shape[0]
#     train_loss = 0
#     cont_loss = 0
#     dis_loss = 0
#     right_num =0
#     right_num1 =0
#     sum = 0
#     matrix = WE.numpy()
#     total_elements = matrix.size
#     missing_elements = np.sum(matrix == 0)
#     missing_rate = missing_elements / total_elements
#     total_prob = 0
#     for batch_idx in range(int(np.ceil(num_X / args.batch_size))):
#         idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X)]
#         mul_X_batch = []
#         for iv, X in enumerate(mul_X):
#             mul_X_batch.append(X[idx].to(device))
#         we = WE[idx].to(device)
#         yt_label_correct = torch.tensor(yt_label_correct)
#         sub_target = yt_label_correct[idx].float().to(device)  # batch_y
#         true_target = torch.tensor(yt_label_clean)[idx].float().to(device)

#         optimizer.zero_grad()
#         word_embedding_train = word_embedding_train.to(device)

#         x_bar_list, yLable_logit,yLable_word_logit, fusion_z, individual_zs, z_word ,individual_zs_p,z_word_list,z_d ,z_word_select_all = model(mul_X_batch,we,word_embedding_train,args.k,epoch,'train')

#         # loss_Cont = 0
#         # for i in range(len(individual_zs)):
#         #     loss_Cont += contrast_loss1(z_word, individual_zs[i],we[:,i],args.q,args.tau)
        
#         # loss_Cont1 = 0
#         # for i in range(len(individual_zs)):
#         #     loss_Cont1 += contrast_loss1(z_word, individual_zs_p[i],we[:,i],args.q,args.tau)
        
#         y_prob = torch.softmax(yLable_logit,dim=1).max(1)[0].sum()

#         total_prob += y_prob.item()

#         loss_CL = -torch.mean(torch.sum(F.log_softmax(yLable_logit, dim=1) * sub_target, dim=1))






#         right_num += np.sum(np.argmax(yLable_logit.detach().cpu().numpy(),axis=1)==np.argmax(true_target.detach().cpu().numpy(),axis=1))
#         # right_num1 += np.sum(np.argmax(yLable_word_logit.detach().cpu().numpy(), axis=1) == np.argmax(true_target.detach().cpu().numpy(),axis=1))


#         loss_fusion = loss_CL


#         loss_fusion.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         train_loss += loss_CL.cpu().item()
#         # cont_loss += loss_Cont.cpu().item()
#     # if (epoch+1)%20==0:
#     #     print(total_prob/num_X)
#     train_loss = train_loss/num_X
#     acc_fea = right_num/num_X
#     acc_word = right_num1/num_X

#     return train_loss,acc_fea,acc_word,cont_loss,dis_loss






# def test(epoch,mul_X_val,yv_label_clean,yt_label_correct,args,model,device,optimizer,loss_model,loss,WE_val,index_array,word_embedding_train):
#     model.eval()
#     num_X_val = mul_X_val[0].shape[0]
#     val_loss = 0
#     right_num=0
#     index_array = torch.randperm(num_X_val)
#     matrix = WE_val.numpy()
#     total_elements = matrix.size
#     # 统计缺失值（0）个数
#     missing_elements = np.sum(matrix == 0)
#     # 计算缺失率
#     missing_rate = missing_elements / total_elements
#     # print(f"测试缺失率: {missing_rate:.2f}")
#     right_num1 = 0

#     with torch.no_grad():
#         for batch_idx in range(int(np.ceil(num_X_val / args.batch_size))):
#             idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X_val)]
#             val_target_clean = np.array(yv_label_clean)
#             val_target_clean = torch.tensor(val_target_clean)
#             # print("idx:",idx)
#             mul_X_batch = []
#             for iv, X in enumerate(mul_X_val):
#                 mul_X_batch.append(X[idx].to(device))
#             we_val = WE_val[idx].to(device)
#             word_embedding_train = word_embedding_train.to(device)
#             sub_target_val = val_target_clean[idx].float().to(device)  # 相当于batch_y
#             x_bar_list_val, yLable_logit,yLable_word_logit, fusion_z_val, individual_zs_val,z_word ,content_style_pair,_,z_d = model(mul_X_batch,we_val,word_embedding_train,args.k,epoch,'val')
#             # val_bat_loss = loss(yLable_logit, sub_target_val, model)
#             val_bat_loss = -torch.mean(torch.sum(F.log_softmax(yLable_logit, dim=1) * sub_target_val, dim=1))
#             val_loss += val_bat_loss.cpu().item()
#             a = np.argmax(yLable_logit.detach().cpu().numpy(), axis=1)
#             b = np.argmax(sub_target_val.detach().cpu().numpy(), axis=1)
#             right_num += np.sum(np.argmax(yLable_logit.detach().cpu().numpy(), axis=1) == np.argmax(sub_target_val.detach().cpu().numpy(), axis=1))
#             sub_target_int1 = np.argmax(yLable_word_logit.detach().cpu().numpy(), axis=1)
#             right_num1 += np.sum(sub_target_int1 == b)
#     # if (epoch + 1) == 100:
#     #     with open(r'D:\python project\MVMLC\新方法 - 副本 - 副本\feature\feature.pkl', 'wb') as f:  # 'wb' 表示以二进制写模式打开文件
#     #         pickle.dump(z_d.detach().numpy(), f)
#     #     with open(r'D:\python project\MVMLC\新方法 - 副本 - 副本\feature\label.pkl', 'wb') as f:  # 'wb' 表示以二进制写模式打开文件
#     #         pickle.dump(z_d.detach().numpy(), f)
#     # print("val:", right_num1 / num_X_val)
#     val_loss = val_loss/num_X_val
#     acc_fea = right_num/num_X_val
#     acc_word = right_num1/num_X_val
#     return val_loss,acc_fea,acc_word,fusion_z_val





def test(epoch, mul_X_val, yv_label_clean, yt_label_correct, args, model, device, optimizer, loss_model, loss, WE_val, index_array, word_embedding_train):
    model.eval()  # 设置模型为评估模式
    num_X_val = mul_X_val[0].shape[0]  # 获取验证集样本数量
    val_loss = 0  # 初始化验证损失
    right_num = 0  # 初始化正确分类的数量
    right_num1 = 0  # 初始化另一个正确分类的数量（用于其他任务）

    # 将数据移动到设备（如 GPU）
    mul_X_batch = [X.to(device) for X in mul_X_val]  # 将所有输入数据移动到设备
    we_val = WE_val.to(device)  # 将词嵌入数据移动到设备
    word_embedding_train = word_embedding_train.to(device)  # 将训练词嵌入移动到设备
    val_target_clean = torch.tensor(yv_label_clean).float().to(device)  # 将标签数据移动到设备

    with torch.no_grad():  # 禁用梯度计算
        # 直接对所有数据进行推理
        x_bar_list_val, yLable_logit, yLable_word_logit, fusion_z_val, individual_zs_val, z_word, content_style_pair, _, z_d,z_word_select_all = model(
            mul_X_batch, we_val, word_embedding_train, args.k, epoch, 'val'
        )

        # 计算损失
        val_bat_loss = -torch.mean(torch.sum(F.log_softmax(yLable_logit, dim=1) * val_target_clean, dim=1))
        val_loss += val_bat_loss.cpu().item()

        # 计算分类准确率
        pred_labels = np.argmax(yLable_logit.detach().cpu().numpy(), axis=1)
        true_labels = np.argmax(val_target_clean.detach().cpu().numpy(), axis=1)
        right_num += np.sum(pred_labels == true_labels)

        # 计算另一个任务的分类准确率
        # pred_word_labels = np.argmax(yLable_word_logit.detach().cpu().numpy(), axis=1)
        # right_num1 += np.sum(pred_word_labels == true_labels)

    # 计算平均损失和准确率
    val_loss = val_loss / num_X_val
    acc_fea = right_num / num_X_val
    acc_word = right_num1 / num_X_val
    # if (epoch + 1) == 100:
    #     print(epoch+1)
    #     with open(r'D:\python project\MVMLC\新方法 - 副本 - 副本\feature\feature_imputaion.pkl', 'wb') as f:  # 'wb' 表示以二进制写模式打开文件
    #         pickle.dump(z_d.detach().cpu().numpy(), f)
    #     with open(r'D:\python project\MVMLC\新方法 - 副本 - 副本\feature\label.pkl', 'wb') as f:  # 'wb' 表示以二进制写模式打开文件
    #         pickle.dump(np.argmax(val_target_clean.detach().cpu().numpy(),axis=1), f)
    #     with open(r'D:\python project\MVMLC\新方法 - 副本 - 副本\feature\knowledge.pkl', 'wb') as f:  # 'wb' 表示以二进制写模式打开文件
    #         pickle.dump(z_word.detach().cpu().numpy(), f)
    #     with open(r'D:\python project\MVMLC\新方法 - 副本 - 副本\feature\feature_no_imputation.pkl', 'wb') as f:  # 'wb' 表示以二进制写模式打开文件
    #         pickle.dump(fusion_z_val.detach().cpu().numpy(), f)
    #     exit()

    return val_loss, acc_fea, acc_word, z_d , z_word

def test_all(epoch, mul_X_val, yv_label_clean, yt_label_correct, args, model, device, optimizer, loss_model, loss, WE_val, index_array, word_embedding_train, word_embedding_test):
    model.eval()
    num_X_val = mul_X_val[0].shape[0]
    val_loss = 0
    val_acc = 0
    right_num1 = 0

    # No batching, we process all data at once
    with torch.no_grad():
        val_target_clean = np.array(yv_label_clean)
        val_target_clean = torch.tensor(val_target_clean)

        # Process all views together
        mul_X_batch = [X.to(device) for X in mul_X_val]
        we_val = WE_val.to(device)
        word_embedding_train = word_embedding_train.to(device)
        sub_target_val = val_target_clean.float().to(device)

        # Forward pass through the model
        x_bar_list_val, target_pre_val, yLable_word, fusion_z_val, individual_zs_val, z_word, content_style_pair,_,_ = model(
            mul_X_batch, we_val, word_embedding_train, args.k
        )

        # Compute the loss
        val_bat_loss = loss(target_pre_val, sub_target_val, model)
        val_loss += val_bat_loss.cpu().item()

        # Calculate accuracy
        pred = np.argmax(target_pre_val.detach().cpu().numpy(), axis=1)
        true = np.argmax(sub_target_val.detach().cpu().numpy(), axis=1)

        sub_target_int1 = np.argmax(z_word.detach().cpu().numpy(), axis=1)
        right_num1 += np.sum(sub_target_int1 == true)

        val_acc += np.sum(pred == true)
    print("val:",right_num1/num_X_val)
    # Return average loss and accuracy
    return val_loss / num_X_val, val_acc / num_X_val, fusion_z_val


def dense_to_onehot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    # 展平的索引值对应相加，然后得到精确索引并修改labels_onehot中的每一个值
    labels_onehot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_onehot

