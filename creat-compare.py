import json
import pandas as pd
import os

# 定义要收集的指标
metrics_to_collect = ['r2', 'mae', 'mse', 'rmse']

# 初始化结果字典
results = {metric: {} for metric in metrics_to_collect}

# 模型列表
models = ['linear_reg', 'decision_tree', 'catboost', 'xgboost', 'nn']

# 收集每个模型的验证指标
for model in models:
    prefix = '1' if model == 'linear_reg' else '2' if model == 'decision_tree' else '3' if model == 'catboost' else '4' if model == 'xgboost' else '5'
    
    # 确定正确的文件名格式
    if model == 'nn':
        metrics_file = f"{prefix}nn_valid_metrics.json"
    else:
        metrics_file = f"{prefix}{model}_valid_metrics.json"
    
    # 读取指标文件
    try:
        with open(metrics_file, 'r') as f:
            model_metrics = json.load(f)
            
        # 收集每个指标
        for metric in metrics_to_collect:
            if metric in model_metrics:
                results[metric][model] = model_metrics[metric]
            else:
                results[metric][model] = None
    except FileNotFoundError:
        print(f"Warning: Metrics file {metrics_file} not found for model {model}")
        for metric in metrics_to_collect:
            results[metric][model] = None

# 创建DataFrame
df_results = pd.DataFrame(results)

# 保存为CSV
df_results.to_csv('model_comparison.csv')

# 创建Markdown摘要
with open('model_metrics_summary.md', 'w', encoding='utf-8') as f:
    f.write("# Model Metrics Comparison\n\n")
    f.write("## Validation Metrics\n\n")
    f.write(df_results.to_markdown())
    
    # 找出最佳模型
    best_model = df_results['r2'].idxmax()
    f.write(f"\n\n## Best Model\n\n")
    f.write(f"Based on R² score, the best model is **{best_model}** with an R² of {df_results['r2'][best_model]:.4f}.\n\n")
    
    # 添加分析
    f.write("## Analysis\n\n")
    f.write("Reasons why this model performed best:\n\n")
    f.write("1. [Add your analysis here]\n")
    f.write("2. [Add your analysis here]\n")
    f.write("3. [Add your analysis here]\n")

print("Model comparison completed and saved to model_comparison.csv and model_metrics_summary.md")
