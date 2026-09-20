# import pickle
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np
# def mytsne(h,y):
#     X = h
#     perplexity = 200
#     tsne = TSNE(n_components=2, random_state=33, perplexity=perplexity)
#     X_tsne = tsne.fit_transform(X)
#     num_label = np.max(y)-np.min(y)+1
#     plt.figure(figsize=(16, 12))
#     if np.min(y) == 1:
#         for i in range(1, num_label + 1):
#             plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=str(i),s=1)
#     else:
#         for i in range(num_label):
#             plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=str(i),s=1)

#     plt.xticks([])
#     plt.yticks([])
#     # plt.savefig('tsne.pdf', dpi=300, format='pdf')
#     plt.show()
#     plt.close()
# # 从 .pkl 文件加载数据
# # feature_imputaion     label       knowledge       feature_no_imputation
# with open(r'D:\python project\MVMLC\新方法 - 副本 - 副本\feature\feature_imputaion.pkl', 'rb') as f:  # 'rb' 表示以二进制读模式打开文件
#     X = pickle.load(f)
# with open(r'D:\python project\MVMLC\新方法 - 副本 - 副本\feature\label.pkl', 'rb') as f:  # 'rb' 表示以二进制读模式打开文件
#     Y = pickle.load(f)
# print(X.shape)

# mytsne(X,Y)


import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def mytsne(h, y, ax, title):
    """
    绘制 t-SNE 图并显示在指定的子图中

    参数:
        h (np.array): 高维数据
        y (np.array): 标签
        ax (matplotlib.axes.Axes): 子图对象
        title (str): 子图标题
    """
    X = h
    perplexity = 20   # Our HW 20; Our fashion 50;MTD HW 10;RANK HW 10;Caltech101 5;Scene15 20;LandUse21 ,NUSWIDEOBJ
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    y = np.argmax(y, axis=1)
    num_label = np.max(y) - np.min(y) + 1

    if np.min(y) == 1:
        for i in range(1, num_label + 1):
            ax.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=str(i), s=1)
    else:
        for i in range(num_label):
            ax.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=str(i), s=1)

    # ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.legend()
dataset = "NUSWIDEOBJ"
missing_rate = 0.5


# 从 .pkl 文件加载数据
with open(r'D:\python project\MVMLC\EXOTIC\tsne_feature_label\feature_{}_{}.pkl'.format(dataset,missing_rate), 'rb') as f:
    X1 = pickle.load(f)
with open(r'D:\python project\MVMLC\EXOTIC\tsne_feature_label\label_{}_{}.pkl'.format(dataset,missing_rate), 'rb') as f:
    Y1 = pickle.load(f)


# with open(r'D:\python project\MVMLC\新方法 - 副本 - 副本\knowledge_HW_0.9.pkl'.format(method,dataset,missing_rate), 'rb') as f:
#     X1 = pickle.load(f)
#     X1 = X1.cpu().numpy()
# with open(r'D:\python project\MVMLC\新方法 - 副本 - 副本\knowledge__label_HW_0.9.pkl'.format(method,dataset,missing_rate), 'rb') as f:
#     Y1 = pickle.load(f)

# 创建画布和单个子图
fig, ax = plt.subplots(figsize=(4, 3))

# 绘制 t-SNE 图
mytsne(X1, Y1, ax, "t-SNE Visualization")

# 显示图像
plt.tight_layout()
plt.savefig(r"D:\python project\MVMLC\新方法 - 副本 - 副本\tsne\tsne_{}_{}.pdf".format(dataset,missing_rate), format='pdf', bbox_inches='tight')
plt.show()
