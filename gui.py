import logging
import json
import os
import sys
import time
import torch
import warnings
import gradio as gr
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional, List

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments

from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar
from pyxtal import pyxtal
import tempfile
import subprocess
import base64

# 将图像文件转换为 base64 编码
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# 将图像路径转换为 base64 编码
img_base64 = img_to_base64("icon.png")
img_html = f'<img src="data:image/png;base64,{img_base64}" alt="LMFlow" style="width: 30%; min-width: 60px; display: block; margin: auto; background-color: transparent;">'

MAX_BOXES = 20

logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")

title = f"""
<h1 align="center">Crystal Synthesis Large Language Model(CSLLM)</h1>
<link rel="stylesheet" href="/path/to/styles/default.min.css">
<script src="/path/to/highlight.min.js"></script>
<script>hljs.highlightAll();</script>

{img_html}

<p>Crystal Synthesis Large Language Model(CSLLM) is an LLM that can not only accurately predict the synthesizability of crystal structures but also recommend synthesis precursors.</p>
<p>This version uses three specialized models: synthesis_llm, method_llm, and precursor_llm.</p>
"""

css = """
#user {                                                                         
    float: right;
    position:relative;
    right:5px;
    width:auto;
    min-height:32px;
    max-width: 60%
    line-height: 32px;
    padding: 2px 8px;
    font-size: 14px;
    background:	#68944A;
    border-radius:5px; 
    margin:10px 0px;
    color:#595959 !important;
}
                                             
#chatbot {                                                                      
    float: left;
    position:relative;
    right:5px;
    width:auto;
    min-height:32px;
    max-width: 60%
    line-height: 32px;
    padding: 2px 8px;
    font-size: 14px;
    background:#2B5A8D;
    border-radius:5px; 
    margin:10px 0px;
    color:#595959 !important;
}
"""

@dataclass
class ChatbotArguments:
    prompt_structure: Optional[str] = field(
        default="{input_text}",
        metadata={
            "help": "prompt structure given user's input text"
        },
    )
    end_string: Optional[str] = field(
        default="\n\n",
        metadata={
            "help": "end string mark of the chatbot's output"
        },
    )
    model_paths: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma-separated paths to the three model directories (synthesis_llm,method_llm,precursor_llm)"
        },
    )
    combination_method: Optional[str] = field(
        default="average",
        metadata={
            "help": "method to combine outputs (average, vote, longest)"
        },
    )

pipeline_name = "inferencer"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

parser = HfArgumentParser((
    ModelArguments,
    PipelineArguments,
    ChatbotArguments,
))
model_args, pipeline_args, chatbot_args = (
    parser.parse_args_into_dataclasses()
)
inferencer_args = pipeline_args

with open(pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

# 解析模型路径
if chatbot_args.model_paths:
    model_paths = chatbot_args.model_paths.split(',')
    if len(model_paths) != 3:
        print("Warning: Expected exactly 3 model paths for synthesis_llm, method_llm, and precursor_llm. Using the provided model path three times.")
        model_paths = [model_args.model_name_or_path] * 3
else:
    # 如果没有指定多个模型路径，则使用命令行提供的模型路径三次
    model_paths = [model_args.model_name_or_path] * 3

# 给模型指定名称
model_names = ["synthesis_llm", "method_llm", "precursor_llm"]
model_path_dict = {name: path for name, path in zip(model_names, model_paths)}

# 函数：加载多个模型
def load_models(model_path_dict, args_template, ds_config, device, dtype):
    """加载多个命名模型"""
    models = {}
    for name, path in model_path_dict.items():
        # 为每个模型创建新的参数
        current_args = ModelArguments(
            model_name_or_path=path,
            lora_model_path=args_template.lora_model_path,
            model_revision=args_template.model_revision,
            model_variant=args_template.model_variant,
            padding_side=args_template.padding_side,
            truncation_side=args_template.truncation_side,
            trust_remote_code=args_template.trust_remote_code,
            model_max_length=args_template.model_max_length,
            # 添加其他必要参数
        )
        
        model = AutoModel.get_model(
            current_args,
            tune_strategy='none',
            ds_config=ds_config,
            device=device,
            torch_dtype=dtype
        )
        models[name] = model
        print(f"Loaded model {name} from: {path}")
    
    return models

# 加载三个模型
models = load_models(
    model_path_dict, 
    model_args, 
    ds_config, 
    pipeline_args.device, 
    torch.float16
)

# 为每个模型创建推理器
inferencers = {}
for name in model_names:
    inferencer = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=DatasetArguments(dataset_path=None),
        pipeline_args=pipeline_args,
    )
    inferencers[name] = inferencer

# We don't need input data, we will read interactively from stdin
data_args = DatasetArguments(dataset_path=None)
dataset = Dataset(data_args)

# Chats
model_display_info = ", ".join([f"{name}({os.path.basename(path)})" for name, path in model_path_dict.items()])
print(f"Using models: {model_display_info}")

end_string = "#"
prompt_structure = "###Human: {input_text}###Assistant:"
token_per_step = 4

def structure_to_str(structure):
    # 转换pymatgen结构为pyxtal结构
    px_structure = pyxtal()
    px_structure._from_pymatgen(structure)
    # 获取空间群信息
    spg_symbol = px_structure.group.number
    
    # 获取晶格参数
    a, b, c, alpha, beta, gamma = [float(i.replace(' ','')) for i in str(px_structure.lattice).split(',')[:-1]]
    lattice_str = f"|{a:.3f},{b:.3f},{c:.3f},{alpha:.2f},{beta:.2f},{gamma:.2f}|"
    # 获取Wyckoff位置信息
    wyckoff_strings = []
    for site in px_structure.atom_sites:
        symbol = site.specie  # 获取元素符号
        letter = site.wp.letter  # 获取Wyckoff位置
        multiplicity = site.wp.multiplicity
        position = site.position
        wyckoff_strings.append(f"({symbol}-{multiplicity}{letter}{position})")

    return f"{spg_symbol} {lattice_str} {'->'.join(wyckoff_strings)}"

def hist2context(hist):
    context = ""
    for query, response in hist:
        context += prompt_structure.format(input_text=query)
        if not (response is None):
            context += response
    return context

# 为每个模型准备不同的提示词
def prepare_model_prompts(structure):
    # 获取结构描述和化学式
    structure_description = structure_to_str(structure)
    formula = structure.formula
    
    prompts = {
        "synthesis_llm": f"Can this material structure be synthesized {structure_description}?",
        "method_llm": f"How can this material structure be synthesized {formula}?",
        "precursor_llm": f"How can this material structure be synthesized {formula}?"
    }
    
    return prompts, structure_description

def combine_outputs(outputs, method="average"):
    """
    合并多个模型的输出
    
    Args:
        outputs: 输出字符串字典
        method: 合并方法 ("average", "vote", "longest")
    
    Returns:
        合并后的输出字符串
    """
    # 将字典转换为列表
    outputs_list = list(outputs.values())
    
    if method == "longest":
        # 返回最长的输出
        return max(outputs_list, key=len)
    
    elif method == "vote":
        # 按词进行投票
        word_lists = [output.split() for output in outputs_list]
        max_len = max(len(words) for words in word_lists)
        
        # 用空字符串填充较短的输出
        for i in range(len(word_lists)):
            if len(word_lists[i]) < max_len:
                word_lists[i].extend([''] * (max_len - len(word_lists[i])))
        
        # 对每个位置的词进行投票
        result = []
        for pos in range(max_len):
            words_at_pos = [word_list[pos] for word_list in word_lists if pos < len(word_list) and word_list[pos]]
            if not words_at_pos:
                break
                
            # 计算每个词的出现次数
            word_counts = {}
            for word in words_at_pos:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
                    
            # 获取出现最多的词
            most_common = max(word_counts.items(), key=lambda x: x[1])[0]
            result.append(most_common)
            
        return ' '.join(result)
    
    else:  # 默认使用average方法
        # 基于字符位置的平均
        min_len = min(len(output) for output in outputs_list)
        result = ""
        
        # 对于每个位置，取最常见的字符
        for i in range(min_len):
            chars = [output[i] for output in outputs_list]
            char_counts = {}
            for char in chars:
                if char in char_counts:
                    char_counts[char] += 1
                else:
                    char_counts[char] = 1
                    
            most_common_char = max(char_counts.items(), key=lambda x: x[1])[0]
            result += most_common_char
            
        # 添加最长输出的剩余部分
        longest_output = max(outputs_list, key=len)
        if len(longest_output) > min_len:
            result += longest_output[min_len:]
            
        return result

def chat_stream_with_structure(structure, history=None):
    """针对结构的流式对话，使用不同的提示词"""
    if history is None:
        history = []

    # 为每个模型准备特定的提示词
    model_prompts, structure_description = prepare_model_prompts(structure)
    
    # 存储每个模型的完整输出
    model_outputs = {name: "" for name in model_names}
    print_index = 0
    
    # 为每个模型准备上下文和输入数据集
    contexts = {}
    input_datasets = {}
    for name, model in models.items():
        prompt = model_prompts[name]
        context = prompt_structure.format(input_text=prompt)
        context_ = context[-model.get_max_length():]
        input_dataset = dataset.from_dict({
            "type": "text_only",
            "instances": [{"text": context_}]
        })
        contexts[name] = context_
        input_datasets[name] = input_dataset
    
    # 打印提示词（用于调试）
    for name, prompt in model_prompts.items():
        print(f"{name} prompt: {prompt}")
    
    # 为每个模型创建生成器
    generators = {}
    for name in model_names:
        generator = inferencers[name].stream_inference(
            context=contexts[name],
            model=models[name],
            max_new_tokens=inferencer_args.max_new_tokens,
            token_per_step=token_per_step,
            temperature=inferencer_args.temperature,
            end_string=end_string,
            input_dataset=input_datasets[name]
        )
        generators[name] = generator
    
    # 跟踪活跃的生成器
    active_generators = set(model_names)
    query = f"Crystal structure analysis: {structure_description}"
    
    while active_generators:
        for name in list(active_generators):
            try:
                response, flag_break = next(generators[name])
                model_outputs[name] = response
                
                if flag_break:
                    active_generators.remove(name)
            except StopIteration:
                active_generators.remove(name)
        
        # 合并当前所有模型的输出
        combined_output = combine_outputs(model_outputs, chatbot_args.combination_method)
        
        # 只返回新内容
        if len(combined_output) > print_index:
            delta = combined_output[print_index:]
            print_index = len(combined_output)
            yield delta, history + [(query, combined_output)]
    
    # 最终合并输出，确保所有内容都返回
    final_combined = combine_outputs(model_outputs, chatbot_args.combination_method)
    if len(final_combined) > print_index:
        delta = final_combined[print_index:]
        yield delta, history + [(query, final_combined)]

def chat_stream(query: str, history=None, structure=None):
    """常规流式对话"""
    if history is None:
        history = []

    # 如果有结构，使用特定的处理方法
    if structure is not None:
        yield from chat_stream_with_structure(structure, history)
        return

    context = hist2context(history)
    context += prompt_structure.format(input_text=query)
    
    # 使用第一个模型处理常规查询
    default_model_name = model_names[0]
    model = models[default_model_name]
    inferencer = inferencers[default_model_name]
    
    context_ = context[-model.get_max_length():]
    input_dataset = dataset.from_dict({
        "type": "text_only",
        "instances": [{"text": context_}]
    })
    
    print(f"Regular query using {default_model_name}: {context_}")
    
    print_index = 0
    for response, flag_break in inferencer.stream_inference(
            context=context_, 
            model=model, 
            max_new_tokens=inferencer_args.max_new_tokens,
            token_per_step=token_per_step, 
            temperature=inferencer_args.temperature,
            end_string=end_string, 
            input_dataset=input_dataset
        ):
        
        delta = response[print_index:]
        seq = response
        print_index = len(response)

        yield delta, history + [(query, seq)]
        if flag_break:
            break

def predict(input, history=None, structure=None): 
    if history is None:
        history = []
    for response, history in chat_stream(input, history, structure):
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="" + query))
            updates.append(gr.update(visible=True, value="" + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.update(visible=False)] * (MAX_BOXES - len(updates))
        yield [history] + updates

def visualize_structure(file):
    file_path = file.name

    if file_path.endswith('.cif'):
        try:
            structure = CifParser(file_path).get_structures()[0]
        except Exception as e:
            return f"Error parsing CIF file: {str(e)}", None
    elif file_path.endswith('POSCAR'):
        try:
            structure = Poscar.from_file(file_path).structure
        except Exception as e:
            return f"Error parsing POSCAR file: {str(e)}", None
    else:
        return "Unsupported file format", None

    # 生成VESTA输入文件
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))
    structure.to(fmt='poscar', filename=temp_file_path)

    # VESTA可执行文件路径（根据实际路径修改）
    vesta_executable = '/home/zhilong666/桌面/VESTA-gtk3/VESTA'

    # 调用VESTA命令行工具打开文件
    try:
        subprocess.Popen([vesta_executable, temp_file_path])
    except Exception as e:
        return f"Failed to open VESTA: {str(e)}", structure

    # 将结构转换为描述性字符串
    structure_description = structure_to_str(structure)
    return f"Analyzing structure: {structure_description}", structure

with gr.Blocks(css=css) as demo:
    gr.HTML(title)
    
    with gr.Accordion("Model Configuration", open=False):
        gr.Markdown(f"Using models: {model_display_info}")
        gr.Markdown(f"Combination method: {chatbot_args.combination_method}")
    
    state = gr.State([])
    current_structure = gr.State(None)  # 存储当前结构
    
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="Q:", elem_id="user"))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="A:", elem_id="chatbot"))

    txt = gr.Textbox(
        show_label=False,
        placeholder="Enter text and press send.",
    )
    button = gr.Button("Send")

    button.click(predict, [txt, state, current_structure], [state] + text_boxes)

    gr.Markdown("# Crystal Structure Analysis")
    file_input = gr.File(label="Upload CIF or POSCAR file")

    def handle_file_upload(file, state):
        structure_description, structure = visualize_structure(file)
        # 对于结构分析，直接开始分析流程
        if structure is not None:
            return structure_description, structure
        else:
            return structure_description, None

    file_input.change(
        fn=handle_file_upload, 
        inputs=[file_input, state], 
        outputs=[txt, current_structure]
    )
    
    # 添加一个分析按钮，用于直接分析上传的结构
    analyze_button = gr.Button("Analyze Structure")
    
    def start_analysis(structure, history):
        if structure is None:
            return history + [("No structure loaded. Please upload a valid structure file first.", "")]
        
        # 直接返回预测结果
        return predict("", history, structure)
    
    analyze_button.click(
        fn=start_analysis,
        inputs=[current_structure, state],
        outputs=[state] + text_boxes
    )

demo.launch(share=True)
