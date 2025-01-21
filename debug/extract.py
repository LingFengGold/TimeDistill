import os
import re
import pandas as pd
import numpy as np

# 定义文件所在目录
directory = '.'  # 假设所有文件都在同一目录下

# 定义用于提取 mse 和 mae 的正则表达式模式
pattern = r"mse:([0-9.]+), mae:([0-9.]+)"

# 存储提取信息的数据列表
data = []

# 提取 mse 和 mae 的函数
def extract_mse_mae(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(pattern, content)
        if matches:
            # 获取最后一次出现的 mse 和 mae
            mse, mae = matches[-1]
            return float(mse), float(mae)
    return None, None

# 处理目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith(".out"):  # 只处理 .out 文件
        parts = filename.split("_")
        if len(parts) >= 11:
            dataset = parts[0]
            pred_len = int(parts[1])
            lr = float(parts[2])
            lr_schedule = parts[3]
            d_model = int(parts[4])
            train_epoch = int(parts[5])
            norm = parts[6]
            alpha = float(parts[7])
            beta = float(parts[8])
            model_t = parts[9]
            method = "_".join(parts[10:]).replace('.out', '')  # 将 `method` 部分整体取出

            mse, mae = extract_mse_mae(os.path.join(directory, filename))
            if mse is not None and mae is not None:
                data.append([dataset, pred_len, lr, lr_schedule, d_model, train_epoch, norm, alpha, beta, model_t, method, mse, mae])

# 将数据转换为 DataFrame，并四舍五入至 5 位有效数字
df = pd.DataFrame(data, columns=['dataset', 'pred_len', 'lr', 'lr_schedule', 'd_model', 'train_epoch', 'norm', 'alpha', 'beta', 'model_t', 'method', 'mse', 'mae'])
df['mse'] = df['mse'].apply(lambda x: round(x, 5))
df['mae'] = df['mae'].apply(lambda x: round(x, 5))

# 按 norm、dataset、pred_len、alpha、beta、lr、lr_schedule、d_model、train_epoch 排序
df_sorted = df.sort_values(by=['norm', 'dataset', 'pred_len', 'alpha', 'beta', 'lr', 'lr_schedule', 'd_model', 'train_epoch'])

# 保存为 CSV 文件
output_file_1 = './results_by_file.csv'
df_sorted.to_csv(output_file_1, index=False)

# 计算每个 norm、dataset、alpha、beta、lr、lr_schedule、d_model、train_epoch、method 的平均 mse 和 mae
df_avg = df.groupby(['norm', 'dataset', 'alpha', 'beta', 'lr', 'lr_schedule', 'd_model', 'train_epoch', 'method']).agg(
    mse_mean=('mse', 'mean'),
    mae_mean=('mae', 'mean'),
    pred_len_count=('pred_len', 'nunique')
).reset_index()

# 筛选出 pred_len 数量少于 4 的组，并将其均值设为 None
df_avg.loc[df_avg['pred_len_count'] < 4, ['mse_mean', 'mae_mean']] = None

# 删除计数列
df_avg.drop(columns=['pred_len_count'], inplace=True)

# 将 mse_mean 和 mae_mean 列确保四舍五入到 5 位小数
df_avg['mse_mean'] = df_avg['mse_mean'].apply(lambda x: round(x, 5) if pd.notnull(x) else x)
df_avg['mae_mean'] = df_avg['mae_mean'].apply(lambda x: round(x, 5) if pd.notnull(x) else x)

# 按 norm、dataset、alpha、beta、lr、lr_schedule、d_model、train_epoch、method 进行排序后保存为 CSV 文件
df_avg_sorted = df_avg.sort_values(by=['norm', 'dataset', 'alpha', 'beta', 'lr', 'lr_schedule', 'd_model', 'train_epoch', 'method'])

# 保存平均结果为另一个 CSV 文件
output_file_2 = './results_average.csv'
df_avg_sorted.to_csv(output_file_2, index=False)
