import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import math

def main():
    print("开始XGBoost模型在完整数据集上的评估...")
    
    # 加载完整数据集
    X_full = pd.read_csv('X_full.csv')
    y_full = pd.read_csv('y_full.csv')
    
    
    # 创建DMatrix对象
    dfull = xgb.DMatrix(X_full)
    
    # 加载训练好的模型
    model = xgb.Booster()
    model.load_model('4xgb_model.json')
    
    # 在完整数据集上评估模型
    y_pred = model.predict(dfull)
    
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
    
    with open('4xgboost_full_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("完整数据集评估指标已保存到 4xgboost_full_metrics.json")

if __name__ == "__main__":
    main()
