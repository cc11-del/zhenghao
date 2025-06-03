import yaml
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import math
import matplotlib.pyplot as plt

# 加载params.yaml文件
with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)
# 获取params.yaml中的决策树超参数
depth = params['catboost']['depth']
learning_rate = params['catboost']['learning_rate']
random_state = params['catboost']['random_state']

def main():
    print("开始CatBoost模型训练...")
    
    # 加载参数
    try:
        with open('params.yaml', 'r', encoding='utf-8') as f:  # 指定utf-8编码
            params = yaml.safe_load(f)
        
        # 获取CatBoost参数
        cb_params = params['catboost']
        print(f"成功加载参数: {cb_params}")
    except Exception as e:
        print(f"加载参数时出错: {e}")
        print("使用默认参数...")
        cb_params = {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'random_state': 42
        }
    
    # 加载60%训练数据
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    
    # 训练CatBoost模型，使用params.yaml中的参数
    model = CatBoostRegressor(
        iterations=cb_params['iterations'],
        learning_rate=cb_params['learning_rate'],
        depth=cb_params['depth'],
        loss_function='RMSE',
        random_seed=cb_params['random_state'],
        verbose=100,  # 每100次迭代输出一次信息
        task_type='CPU'  # 明确指定CPU模式，避免Tcl/Tk问题
    )
    
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
    model.save_model('3catboost_model.cbm')
    print("CatBoost模型已保存到 3catboost_model.cbm")
    
    # 保存评估指标
    metrics = {
        "mse": float(mse),  # 确保值是可序列化的
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "iterations": cb_params['iterations'],
        "learning_rate": cb_params['learning_rate'],
        "depth": cb_params['depth']
    }
    
    with open('3catboost_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("训练指标已保存到 3catboost_metrics.json")
    
    try:
        # 获取特征重要性
        feature_importances = model.get_feature_importance()
        feature_names = X_train.columns
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })
        
        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # 保存特征重要性到CSV
        feature_importance_df.to_csv('3catboost_feature_importance.csv', index=False)
        print("特征重要性已保存到 3catboost_feature_importance.csv")
        
        # 绘制特征重要性图，使用非交互式后端
        plt.switch_backend('agg')  # 使用非交互式后端，避免Tcl/Tk问题
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('CatBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('3catboost_feature_importance.png', dpi=300)
        plt.close()  # 确保关闭图形
        print("特征重要性图已保存到 3catboost_feature_importance.png")
    except Exception as e:
        print(f"生成特征重要性时出错: {e}")
        print("跳过特征重要性生成...")

if __name__ == "__main__":
    main()