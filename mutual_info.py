import torch

def cosine_similarity_matrix(A, B):
    """
    计算两个矩阵之间的余弦相似度矩阵。
    
    参数:
        A (torch.Tensor): 矩阵 A，形状为 (m, d)。
        B (torch.Tensor): 矩阵 B，形状为 (n, d)。
    
    返回:
        S (torch.Tensor): 余弦相似度矩阵，形状为 (m, n)。
    """
    # 归一化矩阵 A 和 B
    A_norm = A / A.norm(dim=1, keepdim=True)  # (m, d)
    B_norm = B / B.norm(dim=1, keepdim=True)  # (n, d)

    print(A_norm)
    print(B_norm)
    
    # 计算余弦相似度矩阵
    S = torch.mm(A_norm, B_norm.T)  # (m, n)
    return S

# 示例数据
A = torch.tensor([[100.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=torch.float32)  # 2 个样本，每个样本 3 维

B = torch.tensor([[1.0, 0.0, 1.0],
                  [2.0, 2.0, 2.0],
                  [0.0, 1.0, 0.0]], dtype=torch.float32)  # 3 个样本，每个样本 3 维

# 计算余弦相似度矩阵
similarity_matrix = cosine_similarity_matrix(A, B)
print("余弦相似度矩阵:\n", similarity_matrix)