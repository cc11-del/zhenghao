import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import math

def main():
    print("开始线性回归模型验证...")
    
    # 加载40%验证数据
    X_valid = pd.read_csv('X_test.csv')
    y_valid = pd.read_csv('y_test.csv')
    
    # 加载训练好的模型
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 在验证集上评估模型
    y_pred = model.predict(X_valid)
    
    # 计算评估指标
    mse = mean_squared_error(y_valid, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    
    print(f"验证集评估结果:")
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
    
    with open('1linear_reg_valid_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("验证指标已保存到 1linear_reg_valid_metrics.json")

if __name__ == "__main__":
    main()