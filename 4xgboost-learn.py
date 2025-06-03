import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import math
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置非交互式后端
import matplotlib.pyplot as plt

# 加载params.yaml文件
with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)
# 获取params.yaml中的决策树超参数
max_depth = params['xgboost']['max_depth']
random_state = params['xgboost']['random_state']

def main():
    print("开始XGBoost模型训练...")
    
    # 加载参数
    try:
        with open('params.yaml', 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        
        # 获取XGBoost参数
        xgb_params = params.get('xgboost', {})
        print(f"成功加载参数: {xgb_params}")
    except Exception as e:
        print(f"加载参数时出错: {e}")
        print("使用默认参数...")
        xgb_params = {}
    
    # 加载60%训练数据
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    
    # 创建DMatrix对象
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # 设置XGBoost参数，优先使用params.yaml中的值，否则使用默认值
    model_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': xgb_params.get('eta', 0.1),
        'max_depth': xgb_params.get('max_depth', 6),
        'subsample': xgb_params.get('subsample', 0.8),
        'colsample_bytree': xgb_params.get('colsample_bytree', 0.8),
        'seed': xgb_params.get('random_state', 42)
    }
    
    # 训练XGBoost模型
    num_rounds = xgb_params.get('num_rounds', 500)
    model = xgb.train(model_params, dtrain, num_rounds)
    
    # 在训练集上评估模型
    y_pred = model.predict(dtrain)
    
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
    model.save_model('4xgb_model.json')
    print("XGBoost模型已保存到 4xgb_model.json")
    
    # 保存评估指标
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "num_rounds": num_rounds,
        "eta": model_params['eta'],
        "max_depth": model_params['max_depth'],
        "subsample": model_params['subsample'],
        "colsample_bytree": model_params['colsample_bytree']
    }
    
    with open('4xgboost_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("训练指标已保存到 4xgboost_metrics.json")
    
    try:
        # 获取特征重要性
        feature_importance = model.get_score(importance_type='weight')
        
        # 创建特征重要性DataFrame
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        })
        
        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # 保存特征重要性到CSV
        feature_importance_df.to_csv('4xgboost_feature_importance.csv', index=False)
        print("特征重要性已保存到 4xgboost_feature_importance.csv")
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('4xgboost_feature_importance.png', dpi=300)
        plt.close()  # 确保关闭图形
        print("特征重要性图已保存到 4xgboost_feature_importance.png")
    except Exception as e:
        print(f"生成特征重要性时出错: {e}")
        # 创建空的特征重要性文件，以满足DVC的要求
        pd.DataFrame(columns=['Feature', 'Importance']).to_csv('4xgboost_feature_importance.csv', index=False)
        print("创建了空的特征重要性文件")

if __name__ == "__main__":
    main()