import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1. 导入数据集
file_path = 'Mental_health_Depression_disorder_Data.csv'
data = pd.read_csv(file_path)

# 2. 处理数据
# 将混合类型的列转换为数值
data['Schizophrenia (%)'] = pd.to_numeric(data['Schizophrenia (%)'], errors='coerce')
data['Bipolar disorder (%)'] = pd.to_numeric(data['Bipolar disorder (%)'], errors='coerce')
data['Eating disorders (%)'] = pd.to_numeric(data['Eating disorders (%)'], errors='coerce')
data['Anxiety disorders (%)'] = pd.to_numeric(data['Anxiety disorders (%)'], errors='coerce')
data['Drug use disorders (%)'] = pd.to_numeric(data['Drug use disorders (%)'], errors='coerce')
data['Depression (%)'] = pd.to_numeric(data['Depression (%)'], errors='coerce')
data['Alcohol use disorders (%)'] = pd.to_numeric(data['Alcohol use disorders (%)'], errors='coerce')

# 删除包含NaN值的行
data = data.dropna()

# 3. 特征选择与目标变量
X = data[['Year', 'Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)',
          'Anxiety disorders (%)', 'Drug use disorders (%)', 'Alcohol use disorders (%)']]
y = data['Depression (%)']

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 创建和训练多元线性回归模型
mlr = LinearRegression()
mlr.fit(X_train_scaled, y_train)

# 7. 预测
y_pred = mlr.predict(X_test_scaled)

# 8. 可视化实际值与预测值的关系
plt.figure(figsize=(14, 8))
plt.scatter(y_test, y_pred, color='blue', edgecolor='w', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Depression (%)')
plt.ylabel('Predicted Depression (%)')
plt.title('Multivariate Linear Regression: Actual vs Predicted Depression (%)')
plt.grid(True)
plt.tight_layout()
plt.show()
