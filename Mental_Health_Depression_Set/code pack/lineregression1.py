import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 加载数据
file_path = 'Mental_health_Depression_disorder_Data.csv'  # 替换为您的文件路径
data = pd.read_csv(file_path, low_memory=False)

# 打印前几行数据以了解数据结构
print(data.head())

# 将相关列转换为数值类型，处理非数值值
for column in ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 'Alcohol use disorders (%)']:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# 确保年份列为数值型
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

# 删除包含 NaN 值的行
data.dropna(subset=['Year'] + ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 'Alcohol use disorders (%)'], inplace=True)

# 打印数据类型以确认转换结果
print(data.dtypes)

# 按年份分组并计算每种疾病的平均百分比
numeric_columns = ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 'Alcohol use disorders (%)']
global_trends = data.groupby('Year')[numeric_columns].mean()

# 检查分组后的数据
print(global_trends.head())

# 准备线性回归模型的数据
years = np.array(global_trends.index).reshape(-1, 1)

# 创建一个图形对象
plt.figure(figsize=(15, 10))

# 定义要分析的心理疾病
disorders = ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 'Alcohol use disorders (%)']

# 对每种心理疾病进行分析
for disorder in disorders:
    # 准备数据
    disorder_rates = global_trends[disorder].values

    # 拟合线性回归模型
    model = LinearRegression()
    model.fit(years, disorder_rates)

    # 获取线性回归模型的系数
    beta_0 = model.intercept_
    beta_1 = model.coef_[0]

    # 预测未来10年的趋势
    future_years = np.arange(global_trends.index[-1] + 1, global_trends.index[-1] + 1 + 10).reshape(-1, 1)
    future_forecast = model.predict(future_years)

    # 打印线性回归公式和预测结果
    print(f"{disorder} Linear Regression Formula: y = {beta_0} + {beta_1}x")
    print("Future Predictions:")
    for year, prediction in zip(future_years.flatten(), future_forecast):
        print(f"Year {year}: {prediction:.4f}%")

    # 合并原始数据和预测数据
    all_years = np.concatenate([years, future_years])
    all_disorder_rates = np.concatenate([disorder_rates, future_forecast])

    # 绘制结果图表
    plt.plot(global_trends.index, disorder_rates, label=f'Observed {disorder}')
    plt.plot(future_years, future_forecast, linestyle='--', label=f'Forecasted {disorder}')

# 添加图表信息
plt.axvline(x=global_trends.index[-1], linestyle='--', color='gray', label='Forecast Start')
plt.title('Observed and Forecasted Global Trends of Mental Health Disorders (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(ticks=np.arange(global_trends.index.min(), global_trends.index.max() + 11, 2), rotation=45)  # 每2年显示一个标签
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例放在图表外面
plt.grid(True)
plt.tight_layout()

# 显示图表
plt.show()
