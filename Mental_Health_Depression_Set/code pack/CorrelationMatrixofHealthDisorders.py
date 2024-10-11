import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = 'Mental_health_Depression_disorder_Data.csv'  # 替换为您的文件路径
data = pd.read_csv(file_path, low_memory=False)

# 将相关列转换为数值类型，处理非数值值
for column in ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 'Alcohol use disorders (%)']:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# 计算相关矩阵
correlation_matrix = data.corr()

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix of Mental Health Disorders (1990-2017)', fontsize=15)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.tight_layout()  # 确保布局紧凑，避免文字被截断
plt.show()



