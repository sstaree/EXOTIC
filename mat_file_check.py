import scipy.io
import numpy as np
import pickle

# 读取 .mat 文件
def read_mat_file(file_path):
    """
    读取 .mat 文件并返回其中的数据。

    参数:
        file_path (str): .mat 文件的路径。

    返回:
        data (dict): 包含 .mat 文件中所有变量的字典。
    """
    # 使用 scipy.io.loadmat 读取 .mat 文件
    data = scipy.io.loadmat(file_path)
    return data

# 示例：读取 .mat 文件并打印内容
if __name__ == "__main__":
    # 替换为你的 .mat 文件路径
    file_path = r"D:\WeChat Files\wxid_mww6ecbue97c22\FileStorage\File\2025-02\3V_Fashion_MV.mat"
    
    # 读取 .mat 文件
    mat_data = read_mat_file(file_path)
    X1 = mat_data['X1']
    X1 = X1.reshape(X1.shape[0], -1)
    X2 = mat_data['X2']
    X2 = X2.reshape(X2.shape[0], -1)
    X3 = mat_data['X3']
    X3 = X3.reshape(X3.shape[0], -1)
    X = [X1,X2,X3]
    Y = mat_data['Y'].T

    num_classes = np.max(Y) + 1

    one_hot_labels = np.eye(num_classes)[Y.reshape(-1)]
    with open('./X.pkl', 'wb') as f:  # 'wb' 表示以二进制写模式打开文件
        pickle.dump(X, f) 
    with open('./Y.pkl', 'wb') as f:  # 'wb' 表示以二进制写模式打开文件
        pickle.dump(one_hot_labels, f)   # 将对象保存到文件中
    print("Mat 文件中的变量：", mat_data.keys())
    
