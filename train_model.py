import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import json

def main():
    # 1. 加载训练集数据
    train_data = pd.read_csv("train_data.csv")
    X_train = train_data[['n_student', 'pretest', 'lunch_encoded']]
    y_train = train_data['posttest']
    
    # 2. 加载测试集数据，用于模型评估
    test_data = pd.read_csv("test_data.csv")
    X_test = test_data[['n_student', 'pretest', 'lunch_encoded']]
    y_test = test_data['posttest']
    
    # 3. 构建并训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. 打印模型参数（英文）
    print("Model coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    
    # 5. 利用测试集进行预测，并计算评估指标
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared:", r2)
    
    # 6. 保存训练好的模型
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # 7. 将评估指标保存到 JSON 文件中
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("模型及评估指标已保存")

if __name__ == "__main__":
    main()