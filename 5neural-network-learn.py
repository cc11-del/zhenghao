import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置非交互式后端
import matplotlib.pyplot as plt
import json
import math
import os
import datetime

# 加载params.yaml文件
with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)
# 获取params.yaml中的决策树超参数
batch_size = params['neural_network']['batch_size']
epochs = params['neural_network']['epochs']
random_state = params['neural_network']['random_state']


def main():
    print("开始神经网络模型训练...")
    
    # 加载参数
    try:
        with open('params.yaml', 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        
        # 获取神经网络参数
        nn_params = params.get('neural_network', {})
        print(f"成功加载参数: {nn_params}")
    except Exception as e:
        print(f"加载参数时出错: {e}")
        print("使用默认参数...")
        nn_params = {}
    
    # 创建TensorBoard日志目录
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # 加载60%训练数据
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 保存scaler以便后续使用
    import pickle
    with open('5nn_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # 从参数文件获取超参数，如果不存在则使用默认值
    learning_rate = nn_params.get('learning_rate', 0.001)
    epochs = nn_params.get('epochs', 100)
    batch_size = nn_params.get('batch_size', 32)
    dropout_rate1 = nn_params.get('dropout_rate1', 0.3)
    dropout_rate2 = nn_params.get('dropout_rate2', 0.2)
    patience = nn_params.get('patience', 20)
    
    # 构建神经网络模型
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate1),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    # 编译模型 - 使用明确的损失函数对象而不是字符串
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),  # 使用对象而不是字符串
        metrics=[keras.metrics.MeanAbsoluteError()]  # 使用对象而不是字符串
    )
    
    # 打印模型摘要
    model.summary()
    
    # 训练模型
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            tensorboard_callback,
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    # 保存模型 - 使用.keras扩展名
    model.save('5nn_model.keras')
    print("神经网络模型已保存到 5nn_model.keras")
    
    # 在训练集上评估模型
    y_pred = model.predict(X_train_scaled).flatten()
    
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
    
    # 保存评估指标
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout_rate1": dropout_rate1,
        "dropout_rate2": dropout_rate2,
        "patience": patience
    }
    
    with open('5nn_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("训练指标已保存到 5nn_metrics.json")
    
    try:
        # 绘制学习曲线
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # 检查指标名称
        metric_names = list(history.history.keys())
        print(f"可用的指标名称: {metric_names}")
        
        # 使用正确的指标名称
        mae_key = 'mean_absolute_error'
        val_mae_key = 'val_mean_absolute_error'
        
        # 如果指标名称不存在，尝试其他可能的名称
        if mae_key not in history.history:
            for key in history.history.keys():
                if 'mae' in key.lower() and not key.startswith('val_'):
                    mae_key = key
                    break
        
        if val_mae_key not in history.history:
            for key in history.history.keys():
                if 'mae' in key.lower() and key.startswith('val_'):
                    val_mae_key = key
                    break
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history[mae_key])
        plt.plot(history.history[val_mae_key])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('5nn_learning_curves.png', dpi=300)
        plt.close()  # 确保关闭图形
        print("学习曲线已保存到 5nn_learning_curves.png")
        
        # 绘制权重直方图
        weights = []
        for layer in model.layers:
            if len(layer.weights) > 0:
                weights.append(layer.weights[0].numpy().flatten())
        
        if weights:  # 确保有权重可用
            all_weights = np.concatenate(weights)
            
            plt.figure(figsize=(10, 6))
            plt.hist(all_weights, bins=50)
            plt.title('Distribution of Neural Network Weights')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig('5nn_weights_histogram.png', dpi=300)
            plt.close()  # 确保关闭图形
            print("权重直方图已保存到 5nn_weights_histogram.png")
        else:
            print("没有找到权重，跳过权重直方图生成")
        
        # 保存训练历史
        pd.DataFrame(history.history).to_csv('5nn_training_history.csv', index=False)
        print("训练历史已保存到 5nn_training_history.csv")
        
        print(f"TensorBoard日志已保存到 {log_dir}")
        print("要查看TensorBoard，请在命令行运行: tensorboard --logdir=logs/fit")
    except Exception as e:
        print(f"生成可视化或保存结果时出错: {e}")
        # 确保至少创建必要的输出文件
        if not os.path.exists('5nn_learning_curves.png'):
            plt.figure()
            plt.savefig('5nn_learning_curves.png')
            plt.close()
            print("创建了空的学习曲线图")
        
        if not os.path.exists('5nn_weights_histogram.png'):
            plt.figure()
            plt.savefig('5nn_weights_histogram.png')
            plt.close()
            print("创建了空的权重直方图")
        
        if not os.path.exists('5nn_training_history.csv'):
            pd.DataFrame().to_csv('5nn_training_history.csv', index=False)
            print("创建了空的训练历史CSV")

if __name__ == "__main__":
    main()
