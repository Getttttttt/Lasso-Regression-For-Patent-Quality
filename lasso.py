import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data = pd.read_csv('FinalRegressionData.csv')

# 移除不需要的列
data = data.drop(['id'], axis=1)

# 设置目标变量和特征变量
X = data.drop('current_cited_by', axis=1)
y = data['current_cited_by']

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用带交叉验证的Lasso模型
lasso_cv = LassoCV(alphas=None, cv=10, max_iter=10000)
lasso_cv.fit(X_train, y_train)

# 查看最佳的alpha值
print(f'Optimal alpha value: {lasso_cv.alpha_}')

# 使用最佳alpha值的Lasso回归进行预测
lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

# 计算R^2值
r2 = r2_score(y_test, y_pred)
print(f'R^2: {r2}')

# 计算MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# 计算RMSE
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# 打印系数
print(f'Coefficients: {lasso.coef_}')

coefficients_df = pd.DataFrame({
    'Feature': X.columns,
    'Unscaled Coefficients (B)': lasso.coef_
})

coefficients_df.to_csv('lasso_coefficients_stats.csv', index=False)

# # 计算包括目标变量的完整数据集的相关性矩阵
# full_data = pd.concat([X, y], axis=1)
# corr_matrix = full_data.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix of Features and Target')
# plt.show()


# # 选择前N个最重要的特征（例如，N=10）
# top_n = 10
# top_features = np.abs(lasso.coef_).argsort()[-top_n:]
# selected_features = X.columns[top_features]

# # 创建包含选定特征和目标变量的DataFrame
# selected_data = pd.concat([X.iloc[:, top_features], y], axis=1)

# # 计算选定特征的相关性矩阵
# corr_matrix_selected = selected_data.corr()

# # 绘制相关性矩阵的热图
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix_selected, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix of Top Features and Target')
# plt.xticks(rotation=45)
# plt.yticks(rotation=45)
# plt.show()


# # 假设X_scaled和y已经准备好了
# alphas, coefs, _ = lasso_path(X_scaled, y, alphas=np.logspace(-6, 6, 200))

# # 绘制系数路径
# plt.figure(figsize=(10, 6))
# for coef_l in coefs:
#     plt.plot(alphas, coef_l, lw=2)
# plt.xscale('log')
# plt.xlabel('Alpha', fontsize=14)
# plt.ylabel('Coefficients', fontsize=14)
# plt.title('Lasso Coefficients and Regularization', fontsize=16)
# plt.axis('tight')
# plt.show()


# #绘制预测真实值
# y_pred_adjusted = np.maximum(y_pred, 0)  # 将预测值小于0的调整为0

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=y_test, y=y_pred_adjusted, alpha=0.6, edgecolor=None, color='blue')
# plt.xlabel('Actual Values', fontsize=14)
# plt.ylabel('Predicted Values', fontsize=14)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# plt.fill_between([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
#                  [y_test.min() * 0.9, y_test.max() * 1.1], color='gray', alpha=0.2)
# plt.title('Actual vs. Predicted Values', fontsize=16)
# plt.show()


# 重新计算并排序特征系数
feature_coef = pd.Series(data=lasso.coef_, index=X.columns).sort_values(key=abs, ascending=False)

# 选择系数最大的10个特征
top_features = feature_coef.head(10)  # 这里top_features已经是一个Series了

# 绘制条形图
plt.figure(figsize=(10, 8))
sns.barplot(x=top_features.values, y=top_features.index, orient='h', palette='coolwarm')
plt.xlabel('Coefficients', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.title('Top 10 Features Selected by Lasso', fontsize=16)
plt.tight_layout()  # 自动调整子图参数
plt.show()

