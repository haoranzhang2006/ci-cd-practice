import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 读取数据，跳过表头问题
data = pd.read_csv('training_data.csv')
print("读取的数据：")
print(data.head())

X = data.iloc[:, 0:1]  # 确保只取第一列作为特征
y = data.iloc[:, 1]    # 第二列作为标签

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 保存模型
joblib.dump(model, 'linear_model.pkl')

# 保存系数到 linear_model.txt
with open('linear_model.txt', 'w', encoding='utf-8') as f:
    f.write(f'Coefficients: {model.coef_[0]}\n')
    f.write(f'Intercept: {model.intercept_}\n')

print("模型训练完成，linear_model.txt 已生成！")