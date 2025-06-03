import yaml

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import math
# 在导入matplotlib之前设置后端为非交互式
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
# 加载params.yaml文件
with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

# 获取params.yaml中的决策树超参数
max_depth = params['decision_tree']['max_depth']
min_samples_split = params['decision_tree']['min_samples_split']
random_state = params['decision_tree']['random_state']

def main():
    print("开始决策树模型训练...")
    
    # 加载60%训练数据
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    

    # 训练决策树模型
    # 可以调整超参数以获得更好的性能
    model = DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 在训练集上评估模型
    y_pred = model.predict(X_train)
    
    # 计算评估指标
    mse = mean_squared_error(y_train, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    
    print(f"训练集评估结果:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R^2: {r2}")
    
    # 保存模型
    with open('2dt_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("决策树模型已保存到 dt_model.pkl")
    
    # 保存评估指标
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split
    }
    
    with open('2decision_tree_train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("训练指标已保存到 decision_tree_train_metrics.json")
    
    # 绘制决策树的前几个节点
    try:
        plt.figure(figsize=(20, 10))
        plot_tree(model, max_depth=3, feature_names=['torque', 'max_power', 'km_driven'], 
                filled=True, rounded=True, fontsize=10)
        plt.savefig('2decision_tree.png', dpi=300, bbox_inches='tight')
        print("决策树图已保存到 2decision_tree.png")
    except Exception as e:
        print(f"警告：无法生成决策树图像: {e}")
        # 即使图像生成失败，也继续执行

if __name__ == "__main__":
    main()