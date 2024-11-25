import h5py
from PIL import Image
import numpy as np

# 打开H5文件
# file = h5py.File('/home/hjj/code/data/BraTS19/data/BraTS19_2013_10_1.h5', 'r')
# file = h5py.File('/home/hjj/code/data/LA/data/0RZDK210BSMWAA6467LU/mri_norm2.h5', 'r')
file = h5py.File('/data1/qyd/code/SSL4MIS/data/ACDC/data/slices/patient001_frame01_slice_1.h5','r')

# 查看H5文件中的数据集名称
print("数据集名称列表：", list(file.keys()))

# 读取特定数据集的内容
data = file['image'][:]  # 替换'dataset_name'为实际的数据集名称
print(data.shape)
print(data)
print(np.max(data), np.min(data))
data_norm = (data-np.min(data))/(np.max(data)-np.min(data))
data_norm = data_norm*255
print(data_norm)
print(np.max(data_norm), np.min(data_norm))
image = data_norm[:,:]   #65?
image = Image.fromarray(image, mode='L')
# 保存图像为文件
# image.save('output1.png')

# 获取数组中的唯一值


# 统计唯一值的数量
# count = len(unique_values)

# image = Image.fromarray(image, mode='L')
# 保存图像为文件
# image.save('output1.png')


# 处理数据
# ...

# 关闭H5文件
file.close()