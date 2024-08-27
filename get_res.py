import pandas as pd
import re

with open('slurm-4329905.out', 'r') as f:
    text = f.read()
# 使用正则表达式从文本中提取预测值和真实值
pattern = r"Output: ?(\w+)?\s*groundtruth answer: ?(\w+)?"

values = re.findall(pattern, text)
print(len(values))
# 将提取的值转换为浮点数，如果值为空，则设置为0
predicted_values = [val[0] if val[0] else 0 for val in values]
groundtruth_values = [
    val[1].strip('K') if val[1] else 0 for val in values
]

# # 将提取的值转换为浮点数
# predicted_values = [float(val) if val else 0 for val in predicted_values]
# groundtruth_values = [
#     float(val.strip('K')) if val else 0 for val in groundtruth_values
# ]
print(len(predicted_values))
print(len(groundtruth_values))
# 创建一个 pandas DataFrame
df = pd.DataFrame({
    'Predicted': predicted_values,
    'Groundtruth': groundtruth_values
})

# 保存到 CSV 文件
csv_path = "answers.csv"
df.to_csv(csv_path, index=False)
