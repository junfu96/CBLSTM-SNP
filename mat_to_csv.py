# 将MG数据集转换成csv文件查看数据集格式
import scipy.io
import pandas as pd

# 1. 读取 mat 文件
mat = scipy.io.loadmat('MG_chaos.mat')

# 2. 查看变量名（建议先打印确认）
print(mat.keys())

# 3. 提取 dataset
data = mat['dataset']  # shape: (1201, 1)

# 4. 转为 DataFrame
df = pd.DataFrame(data, columns=['value'])

# 5. 保存为 CSV
df.to_csv('MG.csv', index=False)

print("转换完成！")