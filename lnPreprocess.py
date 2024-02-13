import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('Regression.csv')  # 请将'路径/'替换为实际文件所在的路径

# 对除了'id'列以外的所有列取对数
for col in df.columns:
    if col != 'id':
        df[col] = np.log1p(df[col])

# 保存转换后的数据到一个新的CSV文件
df.to_csv('Regression_log_transformed.csv', index=False)  # 请将'路径/'替换为你希望保存新文件的路径
