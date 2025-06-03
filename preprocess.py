import pandas as pd

def main():
# 读取数据集
    data = pd.read_csv("encoded_data.csv")  # 修改为你的数据文件路径

# 选择特征列
    data = data[['n_student', 'pretest', 'lunch_encoded','posttest']]  # 选择所有特征列

#保存预处理后的数据
    data.to_csv("preprocessed_data.csv",index=False)
    print("预处理完成,数据已保存到preprocessed_data.csv")

if __name__ == "__main__":
    main()