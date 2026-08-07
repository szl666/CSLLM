from mendeleev import element
import matplotlib.cm as cm 
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib.pyplot import MultipleLocator
import numpy as np
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    LayerIntegratedGradients,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
    TextTemplateInput,
    ProductBaselines,
)
import matplotlib.pyplot as plt

def draw_mendeleev(result,down,up,filename):
    plot_data = result

    # 元素周期表中cell的设置
    # cell的大小
    cell_length = 1
    # 各个cell的间隔
    cell_gap = 0.1
    # cell边框的粗细
    cell_edge_width = 0.5

    # 获取各个元素的原子序数、周期数（行数）、族数（列数）以及绘制数据（没有的设置为0）
    elements = []
    for i in range(1, 119):
        ele = element(i)
        ele_group, ele_period = ele.group_id, ele.period

        # 将La系元素设置到第8行
        if 57 <= i <= 71:
            ele_group = i - 57 + 3
            ele_period = 8
        # 将Ac系元素设置到第9行
        if 89 <= i <= 103:
            ele_group = i - 89 + 3
            ele_period = 9
        elements.append([i, ele.symbol, ele_group, ele_period,
                        plot_data.setdefault(ele.symbol,'None')])

    # 设置La和Ac系的注解标签
    elements.append([None, 'LA', 3, 6, None])
    elements.append([None, 'AC', 3, 7, None])
    elements.append([None, 'LA', 2, 8, None])
    elements.append([None, 'AC', 2, 9, None])

    # 新建Matplotlib绘图窗口
    fig = plt.figure(figsize=(20, 10))
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.sans-serif'] = ['Liberation Sans']
    # x、y轴的范围
    xy_length = (20, 11)

    # 获取YlOrRd颜色条
    my_cmap = cm.get_cmap('RdYlGn')
    # 将plot_data数据映射为颜色，根据实际情况调整
    norm = mpl.colors.Normalize(down, up)
    # 设置超出颜色条下界限的颜色（None为不设置，即白色）
    my_cmap.set_under('None')
    # 关联颜色条和映射
    cmmapable = cm.ScalarMappable(norm, my_cmap)
    # 绘制颜色条
    cb = plt.colorbar(cmmapable, drawedges=False)
    tick_locator = ticker.MaxNLocator(nbins=10)
    cb.locator = tick_locator
    cb.update_ticks()
    # 绘制元素周期表的cell，并填充属性和颜色
    for e in elements:
        ele_number, ele_symbol, ele_group, ele_period, ele_count = e
        # print(ele_number, ele_symbol, ele_group, ele_period, ele_count)

        if ele_group is None:
            continue
        # x, y定位cell的位置
        x = (cell_length + cell_gap) * (ele_group - 1)
        y = xy_length[1] - ((cell_length + cell_gap) * ele_period)

        # 增加 La, Ac 系元素距离元素周期表的距离
        if ele_period >= 8:
            y -= cell_length * 0.5

        # cell中原子序数部位None时绘制cell边框并填充热力颜色
        # 即不绘制La、Ac系注解标签地边框以及颜色填充
        if ele_number:
            if ele_count == 'None':
                fill_color = (1.0, 1.0, 1.0, 1.0)
            else:
                fill_color = my_cmap(norm(-ele_count))
            rect = patches.Rectangle(xy=(x, y),
                                    width=cell_length, height=cell_length,
                                    linewidth=cell_edge_width,
                                    edgecolor='k',
                                    facecolor=fill_color)
            plt.gca().add_patch(rect)

        # 在cell中添加原子序数属性
        plt.text(x + 0.04, y + 0.8,
                ele_number,
                va='center', ha='left',
                #  fontdict={'size': 14, 'color': 'black', 'family': 'Helvetica'})
                fontdict={'size': 14, 'color': 'black'})
        # 在cell中添加元素符号
        plt.text(x + 0.5, y + 0.5,
                ele_symbol,
                va='center', ha='center',
                #  fontdict={'size': 14, 'color': 'black', 'family': 'Helvetica', 'weight': 'bold'})
                fontdict={'size': 14, 'color': 'black', 'weight': 'bold'})
        # 在cell中添加热力值
        if type(ele_count) == float:
            plt.text(x + 0.5, y + 0.12,
                    round(-ele_count,2),
                    va='center', ha='center',
                    #  fontdict={'size': 14, 'color': 'black', 'family': 'Helvetica'})
                    fontdict={'size': 14, 'color': 'black'})
        else:
            plt.text(x + 0.5, y + 0.12,
                    ele_count,
                    va='center', ha='center',
                    #  fontdict={'size': 14, 'color': 'black', 'family': 'Helvetica'})
                    fontdict={'size': 14, 'color': 'black'})

    # x, y 轴设置等比例（1:1）（使cell看起来是正方形）
    # plt.axis('equal')
    # 关闭坐标轴
    plt.axis('off')
    # 裁剪空白边缘
    plt.tight_layout()
    # 设置x, y轴的范围
    plt.ylim(0, xy_length[1])
    plt.xlim(0, xy_length[0])

    # 将图保存为*.svg矢量格式
    plt.savefig(filename,bbox_inches='tight',dpi=1200)
    # 显示绘图窗口
    plt.show()


class LLMExplainer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer = self.load_model()
        self.sv = ShapleyValues(self.model)
        self.sv_llm_attr = LLMAttribution(self.sv, self.tokenizer)
        self.fa = FeatureAblation(self.model)
        self.fa_llm_attr = LLMAttribution(self.fa, self.tokenizer)

    def create_bnb_config(self):
        return BitsAndBytesConfig()

    def load_model(self):
        n_gpus = torch.cuda.device_count()
        max_memory = "100000MB"
        bnb_config = self.create_bnb_config()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={i: max_memory for i in range(n_gpus)},
            offload_folder='output_models/finetune_with_lora_sym_50000',
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=True)
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def generate_response(self, prompt, max_new_tokens=15):
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(model_input["input_ids"], max_new_tokens=max_new_tokens)[0]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response

    def attribute(self, template, values, target, num_trials=3, baselines=None, method='shapley'):
        if baselines:
            inp = TextTemplateInput(
                template=template,
                values=values,
                baselines=baselines
            )
        else:
            inp = TextTemplateInput(
                template=template,
                values=values
            )
        
        if method == 'shapley':
            attr_res = self.sv_llm_attr.attribute(inp, target=target, num_trials=num_trials)
        elif method == 'feature_ablation':
            attr_res = self.fa_llm_attr.attribute(inp, target=target)
        else:
            raise ValueError("Invalid attribution method. Choose 'shapley' or 'feature_ablation'.")
        
        return attr_res

    def save_attribution(self, attr_res, filename):
        torch.save(attr_res, filename)

    def get_total_focus(self, attr_res):
        total_focus = defaultdict(float)
        total_focus['sp'] += float(attr_res.seq_attr[0])
        total_focus['lengths'] += float(attr_res.seq_attr[1])
        total_focus['angles'] += float(attr_res.seq_attr[2])
        atom_num = int((len(attr_res.seq_attr)-3)/2)
        symbol_index = [-2*i for i in range(1,atom_num+1)]
        position_index = [-2*i+1 for i in range(1,atom_num+1)]
        total_focus['symbols'] += np.sum([float(attr_res.seq_attr[i]) for i in symbol_index])
        total_focus['positions'] += np.sum([float(attr_res.seq_attr[i]) for i in position_index])
        return total_focus

    def get_mean_total_focus(self, total_focus, num):
        for key in total_focus:
            total_focus[key] /= num
        return total_focus

# Usage example:
# model_name = "llama_hf/llama-7b-hf"
# explainer = LLMExplainer(model_name)

# response = explainer.generate_response("Can this material structure be synthesized \"225 |6.118,6.118,6.118,90.00,90.00,90.00| (Be-4b[0.5 0.5 0.5])->(In-4a[0. 0. 0.])->(Ru-8c[0.25 0.25 0.25])%\"?")
# print(response)

# template = "input: Can this material structure be synthesized \"{} |{}{}| ({}-{})->({}-{})->({}-{})%\"?"
# values = ["225", "6.118,6.118,6.118","90.00,90.00,90.00", "Be","4b[0.5 0.5 0.5]", "In","4a[0. 0. 0.])", "Ru","8c[0.25 0.25 0.25]"]
# target = 'False'
# attr_res = explainer.attribute(template, values, target, method='shapley')
# explainer.save_attribution(attr_res, 'attr_res_shapley.pt')

# attr_res_fa = explainer.attribute(template, values, target, method='feature_ablation')
# explainer.save_attribution(attr_res_fa, 'attr_res_feature_ablation.pt')

# total_focus = explainer.get_total_focus(attr_res)
# mean_total_focus = explainer.get_mean_total_focus(total_focus, 1)
# explainer.draw_mendeleev(mean_total_focus['symbols'], -0.1, 0.31, 'total_focus_symbols.svg')
# explainer.plot_contribution_values(mean_total_focus)

def aggregate_total_focus(attr_res_all):
    """把所有样本的注意力贡献累加后取平均。"""
    total_focus = defaultdict(float)
    for attr_res in attr_res_all:
        for key, value in split_single_focus(attr_res).items():
            total_focus[key] += value
    return {key: value / len(attr_res_all) for key, value in total_focus.items()}


def split_single_focus(attr_res):
    """单个样本按 空间群/晶格长度/晶格角度/元素符号/原子位置 五部分拆分贡献。"""
    focus = defaultdict(float)
    focus['sp'] = float(attr_res.seq_attr[0])
    focus['lengths'] = float(attr_res.seq_attr[1])
    focus['angles'] = float(attr_res.seq_attr[2])
    atom_num = int((len(attr_res.seq_attr) - 3) / 2)
    symbol_index = [-2 * i for i in range(1, atom_num + 1)]
    position_index = [-2 * i + 1 for i in range(1, atom_num + 1)]
    focus['symbols'] = np.sum([float(attr_res.seq_attr[i]) for i in symbol_index])
    focus['positions'] = np.sum([float(attr_res.seq_attr[i]) for i in position_index])
    return focus


def plot_contribution_values(total_focus, filename='Contribution_value.svg'):
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.sans-serif'] = ['Liberation Sans']
    labels = list(total_focus.keys())
    values = list(total_focus.values())
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values)
    cmap = plt.get_cmap('RdYlGn')
    norm = plt.Normalize(min(values), max(values))
    for bar, color in zip(bars, cmap(norm(values))):
        bar.set_color(color)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Contribution value')
    plt.savefig(filename)
    plt.show()


def split_focus_by_label(attr_res_all, pos_token='\u2581True'):
    """按模型预测的 True/False 分组, 分别给出平均贡献。"""
    total_sym = defaultdict(float)
    total_unsym = defaultdict(float)
    count_sym = count_unsym = 0
    for attr_res in attr_res_all:
        focus = split_single_focus(attr_res)
        if attr_res.output_tokens[0] == pos_token:
            for key, value in focus.items():
                total_sym[key] += value
            count_sym += 1
        else:
            for key, value in focus.items():
                total_unsym[key] += value
            count_unsym += 1
    mean = lambda d, n: {k: v / n for k, v in d.items()} if n else {}
    return mean(total_sym, count_sym), mean(total_unsym, count_unsym)


def focus_by_group(attr_res_all, pos_token='\u2581True'):
    """分别按 空间群 / 元素符号 / 元素+位置 统计平均贡献, True 与 False 各一份。"""
    totals = {label: {key: defaultdict(float) for key in ('sp', 'symbols', 'atoms')}
              for label in ('sym', 'unsym')}
    counts = {label: {key: defaultdict(int) for key in ('sp', 'symbols', 'atoms')}
              for label in ('sym', 'unsym')}
    for attr_res in attr_res_all:
        label = 'sym' if attr_res.output_tokens[0] == pos_token else 'unsym'
        sp = list(attr_res.seq_attr_dict.keys())[0]
        totals[label]['sp'][sp] += float(list(attr_res.seq_attr_dict.values())[0])
        counts[label]['sp'][sp] += 1
        atom_num = int((len(attr_res.seq_attr) - 3) / 2)
        for index in [-2 * i for i in range(1, atom_num + 1)]:
            symbol = list(attr_res.input_tokens)[index]
            position = list(attr_res.input_tokens)[index + 1]
            totals[label]['symbols'][symbol] += float(attr_res.seq_attr[index])
            totals[label]['atoms'][symbol + position] += float(
                attr_res.seq_attr[index] + attr_res.seq_attr[index + 1])
            counts[label]['symbols'][symbol] += 1
            counts[label]['atoms'][symbol + position] += 1
    result = {}
    for label in ('sym', 'unsym'):
        result[label] = {
            'sp': {int(k): v / counts[label]['sp'][k] for k, v in totals[label]['sp'].items()},
            'symbols': {k: v / counts[label]['symbols'][k] for k, v in totals[label]['symbols'].items()},
            'atoms': {k: v / counts[label]['atoms'][k] for k, v in totals[label]['atoms'].items()},
        }
    return result


if __name__ == '__main__':
    import sys

    attr_res_path = sys.argv[1] if len(sys.argv) > 1 else 'attr_res_all.pt'
    attr_res_all = torch.load(attr_res_path)
    plot_contribution_values(aggregate_total_focus(attr_res_all))
    total_focus_sym, total_focus_unsym = split_focus_by_label(attr_res_all)
    grouped = focus_by_group(attr_res_all)
