import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
import skimage.measure as measure

import multiprocessing
import numpy as np
from scipy.stats import norm
import ast
import os

# 定义计算 π 系数的函数
def calculate_mixing_coefficients(i, j, cell_centers, cell_types, num_cell_types):
    """
    根据像素点 (i, j) 到每个细胞中心的距离计算 π 系数。
    """
    distances = np.linalg.norm(cell_centers - np.array([i, j]), axis=1)
    mixing_coefficients = np.exp(-distances)  # 假设 π 与距离成反比
    
    # 按细胞类型分组
    mixing_coefficients_grouped = np.zeros(num_cell_types + 1)
    for k in range(num_cell_types):
        # 找到所有属于第 k 种细胞类型的细胞
        indices = (cell_types == k)
        mixing_coefficients_grouped[k] = np.sum(mixing_coefficients[indices])
    
    # 背景噪音的 π
    mixing_coefficients_grouped[-1] = 0.1  # 背景噪音的 π
    
    # 归一化
    mixing_coefficients_grouped /= mixing_coefficients_grouped.sum()
    
    return mixing_coefficients_grouped

def worker(paneln):
    markern = list(panel['marker'])[paneln]
    os.makedirs(f"./res/{markern}",exist_ok=True)
    
    pixel_values = img[paneln,:,:]
    
    # 初始化参数
    num_cell_types = len(np.unique(cell_types))  # 细胞类型
    num_components = num_cell_types + 1  # 加1表示+背景噪音
    means = np.random.rand(num_components)  # 初始化均值
    variances = np.random.rand(num_components)  # 初始化方差
    
    # EM算法
    max_iter = 100
    tolerance = 1e-6
    for iteration in range(max_iter):
        # E步：计算后验概率
        posterior = np.zeros((image_size[0], image_size[1], num_components))
        for i in range(image_size[0]):
            for j in range(image_size[1]):
                pixel_value = pixel_values[i, j]
                mixing_coefficients = mixing_coefficients_all[i, j]  # 使用已知的 π 系数
                
                # 计算后验概率
                for k in range(num_components):
                    posterior[i, j, k] = mixing_coefficients[k] * norm.pdf(pixel_value, means[k], np.sqrt(variances[k]))
                # print(posterior)
                posterior[i, j, :] /= posterior[i, j, :].sum()  # 归一化后验概率
                # print(posterior)
        
        # M步：更新高斯分布的参数（不更新 π 系数）
        new_means = np.zeros(num_components)
        new_variances = np.zeros(num_components)
        
        for k in range(num_components):
            new_means[k] = np.sum(posterior[:, :, k] * pixel_values) / np.sum(posterior[:, :, k])
            new_variances[k] = np.sum(posterior[:, :, k] * (pixel_values - new_means[k]) ** 2) / np.sum(posterior[:, :, k])
        
        # 检查收敛
        if np.allclose(means, new_means, atol=tolerance) and np.allclose(variances, new_variances, atol=tolerance):
            break
        
        means = new_means
        variances = new_variances
    
        # 最终分类
        classification = np.argmax(posterior, axis=2)
        np.save(f"./res/{markern}/posterior_{iteration}.npy", posterior)
        np.save(f"./res/{markern}/classification_{iteration}.npy", classification)


img = tiff.imread('/mnt/public/IMC/xiaoxu/chang/fullstacks/20220314_B2_ROI1_fullstacks.tiff')
mask = tiff.imread('/mnt/public/IMC/xiaoxu/chang/mask/20220314_B2_ROI1_pred_Probabilities_cell_mask.tiff')

panel = pd.read_csv('/mnt/public/IMC/xiaoxu/chang/IMC_CRC_panel.csv')

anno = pd.read_csv('/mnt/public/IMC/xiaoxu/chang/norm_exp_0623_anno.csv')
anno[anno['ID']=='B2_ROI1']
df = anno[anno['ID']=='B2_ROI1']

# 提取每个细胞的中心点位置
df["Position"] = df["Position"].apply(ast.literal_eval)  # 解析字符串为元组
array_2d = df["Position"].apply(pd.Series).to_numpy() 
cell_centers = array_2d.copy()
image_size = img[10,:,:].shape

# 创建细胞类别到整数的映射
cell_type_mapping = {cell: idx for idx, cell in enumerate(df["SubType"].unique())}
df["Cell_Type_Int"] = df["SubType"].map(cell_type_mapping)
cell_types = np.array(df['Cell_Type_Int'].tolist())

# 预计算所有像素点的 π 系数
mixing_coefficients_all = np.zeros((image_size[0], image_size[1], num_components))
for i in range(image_size[0]):
    for j in range(image_size[1]):
        mixing_coefficients_all[i, j] = calculate_mixing_coefficients(i, j, cell_centers, cell_types, num_cell_types)

with multiprocessing.Pool(processes=8) as pool: 
    results = pool.map(worker, range(16,24))

