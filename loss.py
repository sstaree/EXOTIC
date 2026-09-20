import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCELoss(nn.Module):
    def __init__(self, q=0.7):
        """
        初始化 GCE 损失函数
        :param q: 调节超参数，控制对噪声的鲁棒性，0 < q <= 1
        """
        super(GCELoss, self).__init__()
        assert q > 0 and q <= 1, "q must be in the range (0, 1]"
        self.q = q

    def forward(self, preds, labels):
        """
        计算 GCE 损失 (独热向量格式)
        :param preds: 模型预测输出，大小为 (n, c)，表示每个样本属于 c 个类别的概率
        :param labels: 真实标签 (独热向量格式)，大小为 (n, c)
        :return: 计算得到的 GCE 损失值
        """
        # 获取每个样本对正确类别的预测概率
        probs = torch.sum(preds * labels, dim=1)  # (n)

        # 计算 GCE 损失
        loss = (1 - probs.pow(self.q)) / self.q

        # 取平均损失
        return loss.mean()

class Loss(nn.Module):
    def __init__(self, t, device):
        super(Loss, self).__init__()
        self.t = t
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    # def contrast_loss(self, h0, h1, we1, we2):
    #     mask_miss_inst = we1.mul(we2).bool() # mask the unavailable instances
    #
    #     v1 = h0[mask_miss_inst]
    #     v2 = h1[mask_miss_inst]
    #     n = v1.size(0)
    #     N = 2 * n
    #     if n == 0:
    #         return 0
    #     v1 = F.normalize(v1, p=2, dim=1) #normalize two vectors
    #     v2 = F.normalize(v2, p=2, dim=1)
    #
    #     similarity_mat = torch.matmul(v1, v2.T) / self.t
    #
    #     pos = torch.exp(similarity_mat).diag()
    #     fm = torch.exp(similarity_mat).sum(1)
    #
    #     loss = -torch.log(pos/fm).sum()/n
    #     return loss

    def contrast_loss(self, h0, h1, we1, we2):
        # Step 1: 根据 we1 和 we2 创建掩码，过滤掉缺失的样本
        mask_miss_inst = we1.mul(we2).bool()  # we1 和 we2 同时为 1 时，该样本可用

        # Step 2: 根据掩码提取有效的样本特征
        v1 = h0[mask_miss_inst]
        v2 = h1[mask_miss_inst]

        # Step 3: 获取有效样本的数量
        n = v1.size(0)
        if n == 0:
            return 0  # 如果没有有效的样本，返回 0

        # Step 4: 归一化有效样本特征
        v1 = F.normalize(v1, p=2, dim=1)  # 对向量 v1 做 L2 归一化
        v2 = F.normalize(v2, p=2, dim=1)  # 对向量 v2 做 L2 归一化

        # Step 5: 计算相似度矩阵
        similarity_mat = torch.matmul(v1, v2.T) / self.t  # 计算 v1 和 v2 之间的相似度矩阵并除以温度系数 t

        # Step 6: 计算正样本对的相似度 (对角线元素)
        pos = torch.exp(similarity_mat).diag()  # 提取对角线，表示正样本对的相似度

        # Step 7: 计算每个样本与所有样本的相似度总和
        fm = torch.exp(similarity_mat).sum(1)  # 对每个样本的相似度总和

        # Step 8: 计算对比损失
        loss = -torch.log(pos / fm).sum() / n  # 计算负对数损失并平均
        return loss

    def wmse_loss(self, input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        ret = torch.mean(ret)
        return ret
    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        
        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res


class My_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.mm(x,y.t(), 2))


class DistillLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(DistillLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.mask = self.mask_correlated_clusters(class_num).cuda()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        c = F.normalize(c, dim=1)
        sim = c @ c.T / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels) / N

        return loss
    

    import torch
import torch.nn.functional as F

def graph_loss(features, labels, temperature=0.1, lambda_intra=1.0, lambda_inter=1.0):
    """
    Graph-based loss for multi-class classification
    
    Args:
        features (Tensor): Feature vectors (batch_size x feature_dim)
        labels (Tensor): One-hot encoded labels (batch_size x num_classes)
        temperature (float): Scaling factor for similarities
        lambda_intra (float): Intra-class loss weight
        lambda_inter (float): Inter-class loss weight
    
    Returns:
        Tensor: Computed graph loss
    """
    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(features, features.t()) / temperature
    
    # Create class mask (True for same-class pairs)
    class_mask = torch.mm(labels, labels.t()).bool()
    
    # Intra-class loss (pull same-class samples together)
    intra_loss = -sim_matrix[class_mask].mean() if class_mask.any() else 0
    
    # Inter-class loss (push different-class samples apart)
    inter_loss = sim_matrix[~class_mask].mean() if (~class_mask).any() else 0
    
    # Combine losses
    total_loss = lambda_intra * intra_loss + lambda_inter * inter_loss
    
    return total_loss