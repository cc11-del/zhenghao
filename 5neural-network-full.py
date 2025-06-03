import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import math
import pickle

def main():
    print("开始神经网络模型在完整数据集上的评估...")
    
    # 加载完整数据集
    X_full = pd.read_csv('X_full.csv')
    y_full = pd.read_csv('y_full.csv')
    
    # 加载标准化器
    with open('5nn_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # 标准化特征
    X_full_scaled = scaler.transform(X_full)
    
    # 加载训练好的模型 - 使用.keras扩展名
    model = keras.models.load_model('5nn_model.keras')
    
    # 在完整数据集上评估模型
    y_pred = model.predict(X_full_scaled).flatten()
    
    # 计算评估指标
    mse = mean_squared_error(y_full, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_full, y_pred)
    r2 = r2_score(y_full, y_pred)
    
    print(f"完整数据集评估结果:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    
    # 保存评估指标
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    with open('5nn_full_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("完整数据集评估指标已保存到 5nn_full_metrics.json")

if __name__ == "__main__":
    main()