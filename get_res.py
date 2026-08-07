"""从 LMFlow evaluation 的 slurm 日志抽取预测结果并存成 CSV。

用法: python get_res.py slurm-xxxxx.out [answers.csv]

注意: LMFlow 若以 --answer_type math 运行, 日志里的 "predicted answer:" 是从模型续写中
抠出来的数字, 其 accuracy 恒为 0, 不可使用; 真实预测在 "prediction=" 文本的
"Output: True/False" 中。解析逻辑见 result_analyze.get_res_classify。
"""
import sys

import pandas as pd

from result_analyze import get_res_classify


def main(slurm_name, csv_path='answers.csv'):
    structures, predicted_values, groundtruth_values = get_res_classify(slurm_name)
    df = pd.DataFrame({
        'Structure': structures,
        'Predicted': predicted_values,
        'Groundtruth': groundtruth_values
    })
    df.to_csv(csv_path, index=False)
    print(f'{len(df)} 条结果写入 {csv_path}')
    return df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
