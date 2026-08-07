import ast
import re
import jellyfish
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_jaro_winkler_distance(text1, text2):
    # 计算Jaro-Winkler距离
    distance = jellyfish.jaro_winkler_similarity(text1, text2)
    return distance
    
_Q_KEY, _P_KEY, _G_KEY = "question=", "prediction=", "groundtruth answer:"
_QUOTED_RE = re.compile(r'"(.*)"\s*\?', re.S)          # 取问句引号里的结构/化学式
_OUTPUT_RE = re.compile(r"Output\s*[:：]\s*(True|False)\b", re.I)
_BARE_RE = re.compile(r"\b(True|False)\b", re.I)


def parse_slurm_blocks(slurm_name):
    """逐行状态机解析日志, 返回 [(question, prediction, groundtruth), ...]"""
    records, cur = [], {}
    with open(slurm_name, errors="replace") as f:
        for line in f:
            s = line.strip()
            try:
                if s.startswith("batch_index"):
                    cur = {}
                elif s.startswith(_Q_KEY):
                    cur["q"] = ast.literal_eval(s[len(_Q_KEY):])
                elif s.startswith(_P_KEY):
                    cur["p"] = ast.literal_eval(s[len(_P_KEY):])
                elif s.startswith(_G_KEY):
                    cur["g"] = ast.literal_eval(s[len(_G_KEY):].strip())
                    if "q" in cur and "p" in cur:
                        records.extend(zip(cur["q"], cur["p"], cur["g"]))
                    cur = {}
            except (ValueError, SyntaxError):
                cur = {}
    return records


def extract_formula(question):
    """从问句里取引号内的结构串/化学式, 去掉 prompt 前缀和结尾的 %。"""
    q = question[len("input: "):] if question.startswith("input: ") else question
    m = _QUOTED_RE.search(q)
    s = m.group(1) if m else q
    return s.strip().rstrip("%").strip()


def _clean(text):
    return text.replace("Output: ", "").replace("output: ", "")

def get_res(slurm_name,is_str=False,Type=None):
    records = parse_slurm_blocks(slurm_name)
    formulas = [extract_formula(q) for q, p, g in records]
    predict_values = [_clean(p) for q, p, g in records]
    true_values = [_clean(g) for q, p, g in records]
    predict_values_final = []; true_values_final = [];formulas_final = []
    if not is_str:
        if Type:
            for i,j,k in zip(predict_values, true_values,formulas):
                try:
                    stop_index = i.index(", 'operations'")
                    cut_str = i[:stop_index]+'}'
                    predict_values_final.append(eval(cut_str)[Type])
                    true_values_final.append(eval(j)[Type])
                    formulas_final.append(k)
                except:
                    pass
        else:
            for i,j,k in zip(predict_values, true_values,formulas):
                try:
                    predict_values_final.append([i])
                    true_values_final.append([j])
                    formulas_final.append(k)
                except:
                    pass
    else:
        predict_values_final = [i.replace(' ','') for i in predict_values]
        true_values_final = [i.replace(' ','') for i in true_values]
        formulas_final = formulas
    return formulas_final, predict_values_final,true_values_final

def get_res_new(slurm_name,is_str=False,Type=None):
    records = parse_slurm_blocks(slurm_name)
    formulas = [extract_formula(q) for q, p, g in records]
    predict_values = [_clean(p) for q, p, g in records]
    true_values = [_clean(g) for q, p, g in records]
    predict_values_final = []; true_values_final = [];formulas_final = []
    if not is_str:
        if Type:
            for i,j,k in zip(predict_values, true_values,formulas):
                try:                
                    stop_index = i.index(", 'operations'")
                    cut_str = i[:stop_index]+'}'
                    predict_values_final.append(eval(cut_str)[Type])
                    true_values_final.append(eval(j)[Type])
                    formulas_final.append(k)
                except:
                    pass
        else:
            for i,j,k in zip(predict_values, true_values,formulas):
                try:
                    predict_values_final.append([i])
                    true_values_final.append([j])
                    formulas_final.append(k)
                except:
                    pass
    else:
        predict_values_final = [i.replace(' ','') for i in predict_values]
        true_values_final = [i.replace(' ','') for i in true_values]
        formulas_final = formulas
    predict_values_final1 = []; true_values_final1 = [];formulas_final1 = []
    for i,j,k in zip(predict_values_final, true_values_final,formulas_final):
        try:
            index = i[0].split('\n')[1].index(']')
            predict_values_final1.append(eval(i[0].split('\n')[1][:index+1]))
            true_values_final1.append(eval(j[0]))
            formulas_final1.append(k.replace("']",''))
        except Exception:
            print('[get_res_new] 解析失败:', repr(i[0])[:120])
    return formulas_final1, predict_values_final1,true_values_final1

def get_res_classify(slurm_name, keep_unparsed=False):
    """可合成性二分类任务(Can this material structure be synthesized ...? -> True/False)。"""
    structures, preds, truths = [], [], []
    n_block = n_bare = n_drop = 0
    for q, p, g in parse_slurm_blocks(slurm_name):
        n_block += 1
        m = _OUTPUT_RE.search(p)
        if m is None:
            m = _BARE_RE.search(p)
            n_bare += m is not None
        lab = m.group(1).capitalize() if m else None
        if lab is None and not keep_unparsed:
            n_drop += 1
            continue
        structures.append(extract_formula(q))
        preds.append(lab)
        truths.append(str(g).strip().capitalize())
    print(f'[get_res_classify] {slurm_name}: 日志 {n_block} 条 -> 返回 {len(preds)} 条'
          f'(裸 True/False 兜底 {n_bare} 条, 丢弃未解析 {n_drop} 条)')
    if n_drop:
        print(f'  注意: 已丢弃 {n_drop} 条未解析样本, 指标分母不含它们; '
              f'如需按"未解析算错"的严格口径, 用 keep_unparsed=True')
    return structures, preds, truths


def eval_classify(structures, predict_values, true_values, pos_label='True',
                  save_name=None, plot=False):
    """算混淆矩阵与各项指标, 返回 (三列表格 DataFrame, 指标 dict)"""
    df = pd.DataFrame({'structure': structures,
                       'ground_truth': true_values,
                       'prediction': predict_values})
    ok = df['prediction'].notna()
    y_true, y_pred = df.loc[ok, 'ground_truth'], df.loc[ok, 'prediction']
    labels = [pos_label, 'False' if pos_label == 'True' else 'True']
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    kw = dict(pos_label=pos_label, zero_division=0)
    m = {'n_total': len(df), 'n_scored': int(ok.sum()), 'n_unparsed': int((~ok).sum()),
         'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
         'accuracy': accuracy_score(y_true, y_pred),
         'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
         'mcc': matthews_corrcoef(y_true, y_pred),
         'precision': precision_score(y_true, y_pred, **kw),
         'recall': recall_score(y_true, y_pred, **kw),
         'f1': f1_score(y_true, y_pred, **kw),
         'specificity': tn / (tn + fp) if tn + fp else float('nan'),
         'npv': tn / (tn + fn) if tn + fn else float('nan')}
    print(f"覆盖 {m['n_scored']}/{m['n_total']} (未解析 {m['n_unparsed']})")
    print('混淆矩阵 (行=真值, 列=预测)')
    print(f"              pred {pos_label:<6} pred other")
    print(f"  true {pos_label:<8}{tp:>8}{fn:>11}")
    print(f"  true other  {fp:>8}{tn:>11}")
    for k in ('accuracy', 'balanced_accuracy', 'mcc', 'precision', 'recall',
              'f1', 'specificity', 'npv'):
        print(f'  {k:<18}= {m[k]:.4f}')
    if save_name:
        df.to_csv(save_name, index=False)
        print('表格写出', save_name)
    if plot:
        plt.figure(figsize=(4.5, 4))
        sns.heatmap([[tp, fn], [fp, tn]], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=[f'pred {labels[0]}', f'pred {labels[1]}'],
                    yticklabels=[f'true {labels[0]}', f'true {labels[1]}'])
        plt.tight_layout()
        if save_name:
            plt.savefig(str(save_name).rsplit('.', 1)[0] + '_cm.svg')
        plt.show()
    return df, m

from collections import Counter
import matplotlib.cm as cm
def draw_pie(element_counts,save_name,remove_0_key=True,preflix=None):
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.sans-serif'] = ['Liberation Sans']
    if remove_0_key:
        del element_counts[0]
    element_counts = dict(sorted(element_counts.items(), key=lambda x: x[1], reverse=True))
    sizes = list(element_counts.values())
    sizes = [abs(i) for i in sizes]
    labels = list(element_counts.keys())
    colors = cm.Set2.colors[:len(labels)]
    fig, ax = plt.subplots(figsize=(18, 6))  # Keeping a manageable figure size
    edge_props = {'edgecolor': 'grey', 'linewidth': 1, 'alpha': 0.6} 
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', colors=colors, wedgeprops=edge_props)
    for i, text in enumerate(autotexts):
        if sizes[i] < 5:  # Place text outside for smaller slices
            text.set_visible(False)
            angle = (wedges[i].theta1 + wedges[i].theta2) / 2  # Get the mid-angle of the wedge
            # Calculate new text positions based on the angle and the pie radius
            x = np.cos(np.radians(angle)) * 1.1  # 设置位置
            y = np.sin(np.radians(angle)) * 1.1
            ax.text(x, y, f"{sizes[i]}%", ha='center', va='center')  # 将百分比放在外面
        else:  # Display the percentage inside the wedge for larger slices
            text.set_visible(True)
    legend = ax.legend(wedges, labels, title=preflix,loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    legend.get_frame().set_edgecolor('none') 
    plt.savefig(save_name)
    plt.show()

def cal_precursor_res(predict_values,true_values):
    max_distances = []
    for pre_res,true_res in zip(predict_values,true_values):
        distances = []
        for pre_res_single in pre_res:
            distances.append(max([calculate_jaro_winkler_distance(pre_res_single, true_res_single) for true_res_single in true_res]))
        max_distances.append(distances)
    mean_max_distances = [np.mean(i) for i in max_distances]
    success_rate = len([i for i in mean_max_distances if i==1])/len(mean_max_distances)
    return np.mean(mean_max_distances),success_rate,mean_max_distances

def cal_precursor_res_all(formulas, predict_values,true_values, all_precursors,save_name):
    max_distances = [];true_res_len = []
    for formula, pre_res,true_res in zip(formulas,predict_values,true_values):
        try:
            distances = []
            for pre_res_single in pre_res:
                distances.append(max([calculate_jaro_winkler_distance(pre_res_single, true_res_single) for true_res in all_precursors[formula] for true_res_single in true_res]))
            max_distances.append(distances)
            true_res_len.append(len(true_res))
        except:
            pass
    mean_max_distances = [np.mean(i) for i in max_distances]
    success_rate = len([i for i in mean_max_distances if i==1])/len(mean_max_distances)
    miss_redun = [len(i)-j for i,j in zip(max_distances,true_res_len)]
    miss_redun_len = len([i for i in miss_redun if i !=0])
    error_lengths = []
    for max_distance,miss_red in zip(max_distances,miss_redun):
        if miss_red==0:
            error_lengths.append(len(max_distance)-max_distance.count(1.0))
    miss_redun_counts = Counter(miss_redun)
    error_lengths_counts = Counter(error_lengths)
    success_counts = {'Success':len([i for i in mean_max_distances if i==1]),'Precursor error':len(mean_max_distances)-miss_redun_len-len([i for i in mean_max_distances if i==1]),'Precursor missing/excess':miss_redun_len}
    draw_pie(success_counts,f'success_counts_{save_name}.svg',remove_0_key=False)
    draw_pie(miss_redun_counts,f'miss_redun_counts_{save_name}.svg',preflix='Number of missing/excess precursors')
    draw_pie(error_lengths_counts,f'error_lengths_counts_{save_name}.svg',preflix='Number of wrong precursors')
    return np.mean(mean_max_distances),success_rate,mean_max_distances,max_distances,true_res_len,miss_redun_counts,error_lengths_counts
