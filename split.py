import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # 读取预处理后的数据
    data = pd.read_csv("preprocessed_data.csv")
    
    # 分离自变量（X）和目标变量（y）
    X = data[['n_student', 'pretest', 'lunch_encoded']]
    y = data['posttest']
    
    # 按 60% 训练集和 40% 测试集比例划分（设置 random_state=42 保证结果可复现）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)
    
    # 合并训练集和测试集数据
    train_data = X_train.copy()
    train_data['posttest'] = y_train
    test_data = X_test.copy()
    test_data['posttest'] = y_test
    
    # 保存划分后的数据
    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)
    print("数据划分完成：训练集已保存为 train_data.csv,测试集已保存为 test_data.csv")

if __name__ == "__main__":
    main()