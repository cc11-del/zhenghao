import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import math

def main():
    print("开始线性回归模型训练...")
    
    # 加载60%训练数据
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    
    
    # 训练线性回归模型
    model = LinearRegression()
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
    print(f"R2: {r2}")
    
    # 保存模型
    with open('1model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("线性回归模型已保存到 model.pkl")
    
    # 保存评估指标
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_)
    }
    
    with open('1linear_reg_train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("训练指标已保存到 linear_reg_train_metrics.json")

if __name__ == "__main__":
    main()