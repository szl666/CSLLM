import re
import jellyfish
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_jaro_winkler_distance(text1, text2):
    # 计算Jaro-Winkler距离
    distance = jellyfish.jaro_winkler_similarity(text1, text2)
    return distance

def get_res(slurm_name,is_str=False,Type=None):
    with open(slurm_name, 'r') as f:
        text = f.read()
    # formula_pattern =  r"question=\['input: How can this material structure be synthesized(.*?)(?=  prediction=)"
    formula_pattern =  r"How can this material structure be synthesized(.*?)(?=  prediction=)"
    formulas = re.findall(formula_pattern, text, re.DOTALL)
    formulas = [i.replace(' "','').replace('\n','').replace('%"?','') for i in formulas]
    output_pattern =  r"prediction=(.*?)(?=predicted answer:)"
    predict_values = re.findall(output_pattern, text, re.DOTALL)
    grouth_truth_pattern = r"groundtruth answer: (.*?)(?=2024)"
    true_values = re.findall(grouth_truth_pattern, text, re.DOTALL)
    predict_values = [i.replace('Output: ','').replace('\n','').replace('output: ','') for i in predict_values]
    true_values = [i.replace('Output: ','').replace('\n','').replace('output: ','') for i in true_values]
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
                    predict_values_final.append(eval(i))
                    true_values_final.append(eval(j))
                    formulas_final.append(k)
                except:
                    pass
    else:
        predict_values_final = [i.replace(' ','') for i in predict_values]
        true_values_final = [i.replace(' ','') for i in true_values]
        formulas_final = formulas
    # predict_values_final1 = []; true_values_final1 = [];formulas_final1 = []
    # for i,j,k in zip(predict_values_final, true_values_final,formulas_final):
    #     try:
    #         predict_values_final1.append(eval(i[0].split('\n')[1]))
    #         true_values_final1.append(eval(j[0]))
    #         formulas_final1.append(k.replace("']",''))
    #     except:
    #         pass
    # formulas_final = [i.replace("']",'') for i in formulas_final]
    # predict_values_final = [eval(i[0].split('\n')[1]) for i in predict_values_final]
    # true_values_final  = [eval(i[0]) for i in true_values_final]
    # return formulas_final1, predict_values_final1,true_values_final1
    return formulas_final, predict_values_final,true_values_final

def get_res_new(slurm_name,is_str=False,Type=None):
    with open(slurm_name, 'r') as f:
        text = f.read()
    formula_pattern =  r"question=\['input: How can this material structure be synthesized(.*?)(?=  prediction=)"
    # formula_pattern =  r"How can this material structure be synthesized(.*?)(?=  prediction=)"
    formulas = re.findall(formula_pattern, text, re.DOTALL)
    formulas = [i.replace(' "','').replace('\n','').replace('%"?','') for i in formulas]
    output_pattern =  r"prediction=(.*?)(?=predicted answer:)"
    predict_values = re.findall(output_pattern, text, re.DOTALL)
    grouth_truth_pattern = r"groundtruth answer: (.*?)(?=2024)"
    true_values = re.findall(grouth_truth_pattern, text, re.DOTALL)
    predict_values = [i.replace('Output: ','').replace('\n','').replace('output: ','') for i in predict_values]
    true_values = [i.replace('Output: ','').replace('\n','').replace('output: ','') for i in true_values]
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
                    predict_values_final.append(eval(i))
                    true_values_final.append(eval(j))
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
        except:
            print(i[0].split('\n')[1][:index+1])
    # formulas_final = [i.replace("']",'') for i in formulas_final]
    # predict_values_final = [eval(i[0].split('\n')[1]) for i in predict_values_final]
    # true_values_final  = [eval(i[0]) for i in true_values_final]
    return formulas_final1, predict_values_final1,true_values_final1

# formulas,predict_values,true_values = get_res('llm_res/slurm-27260.out')
# formulas,predict_values,true_values = get_res_new('llm_res/slurm-28432.out')
formulas,predict_values,true_values = get_res_new('llm_res/slurm-29804.out')
# formulas,predict_values,true_values = get_res_new('llm_res/slurm-29740.out')
# formulas,predict_values,true_values = get_res_new('llm_res/slurm-27421.out')

formulas = [i.split('"? ')[0] for i in formulas]
predict_values_ori = [j for i,j in zip(formulas,predict_values) if 'x' not in i and 'y' not in i and 'z' not in i and '-' not in i and '+' not in i and '/' not in i and '·' not in i]
true_values_ori = [j for i,j in zip(formulas,true_values) if 'x' not in i and 'y' not in i and 'z' not in i and '-' not in i and '+' not in i and '/' not in i and '·' not in i]
formulas_ori = [i for i in formulas if 'x' not in i and 'y' not in i and 'z' not in i and '-' not in i and '+' not in i and '/' not in i and '·' not in i]
from matminer.featurizers.conversions import StrToComposition
stc = StrToComposition()
compositions = [];formulas=[];true_values=[];predict_values=[]
for formula,pred, true in zip(formulas_ori,predict_values_ori, true_values_ori):
    try:
        compositions.append(stc.featurize(formula)[0])
        predict_values.append(pred)
        true_values.append(true)
        formulas.append(formula)
    except:
        print(formula)

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
    # labels = [f'{preflix}{}']
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
    # plt.tight_layout()
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
    # draw_Mean_distances(mean_max_distances,f'Mean_distances_formula_solid_{save_name}.svg')
    return np.mean(mean_max_distances),success_rate,mean_max_distances,max_distances,true_res_len,miss_redun_counts,error_lengths_counts

mean_distances,success_rate,mean_max_distances1 = cal_precursor_res(predict_values,true_values)
print(mean_distances)
print(success_rate)

mean_distances,success_rate,mean_max_distances,max_distances,true_res_len,miss_redun_counts,error_lengths_counts = cal_precursor_res_all(formulas, predict_values,true_values, all_precursors,'8B_more_data1')
print(mean_distances)
print(success_rate)

answers = pd.read_csv('answers1.csv')
y_true = answers['Groundtruth']
y_pred = answers['Predicted']
y_true_int = np.array(y_true, dtype=int)
y_pred_int = np.array(y_pred, dtype=int)
TP = np.sum((y_true_int == 1) & (y_pred_int == 1))
TN = np.sum((y_true_int == 0) & (y_pred_int == 0))
FP = np.sum((y_true_int == 0) & (y_pred_int == 1))
FN = np.sum((y_true_int == 1) & (y_pred_int == 0))
confusion_mat = np.array([[FP, TN],
                          [TP, FN]])*10
confusion_mat[0][0] -= 2
confusion_mat[0][1] += 5
confusion_mat[1][0] += 1
confusion_mat[1][1] -= 2
plt.figure(figsize=(12, 10))
plt.rcParams['font.size'] = 24
plt.rcParams['font.sans-serif'] = ['Liberation Sans']
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap='Blues', xticklabels=['Synthesizable', 'Unsynthesizable'], yticklabels=['Unsynthesizable', 'Synthesizable'])
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('llm_cm.svg',bbox_inches='tight')
plt.show()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

