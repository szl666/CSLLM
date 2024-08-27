import logging
import json
import os
import sys
import time
# sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import torch
import warnings
import gradio as gr
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

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

<p>Crystal Synthesis Large Language Model(CSLLM) is an LLM  that can not only accurately predict the synthesizability of crystal structures but also recommend synthesis precursors.</p>

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

with open (pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

model = AutoModel.get_model(
    model_args,
    tune_strategy='none',
    ds_config=ds_config,
    device=pipeline_args.device,
    torch_dtype=torch.float16
)

# We don't need input data, we will read interactively from stdin
data_args = DatasetArguments(dataset_path=None)
dataset = Dataset(data_args)

inferencer = AutoPipeline.get_pipeline(
    pipeline_name=pipeline_name,
    model_args=model_args,
    data_args=data_args,
    pipeline_args=pipeline_args,
)

# Chats
model_name = model_args.model_name_or_path
if model_args.lora_model_path is not None:
    model_name += f" + {model_args.lora_model_path}"

end_string = "#"
prompt_structure = "###Human: {input_text}###Assistant:"


# token_per_step = 4

def structure_to_str(structure):
    # 转换pymatgen结构为pyxtal结构
    px_structure = pyxtal()
    px_structure._from_pymatgen(structure)
    # 获取空间群信息
    # spg_symbol = structure.get_space_group_number()
    spg_symbol = px_structure.group.number
    
    # 获取晶格参数
    a, b, c,alpha, beta, gamma = [float(i.replace(' ','')) for i in str(px_structure.lattice).split(',')[:-1]]
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

def chat_stream(query: str, history=None, **kwargs):
    if history is None:
        history = []

    context = hist2context(history)
    print_index = 0
    context += prompt_structure.format(input_text=query)
    context_ = context[-model.get_max_length():]
    input_dataset = dataset.from_dict({
        "type": "text_only",
        "instances": [{"text": context_}]
    })
    print(context_)
    for response, flag_break in inferencer.stream_inference(context=context_, model=model, max_new_tokens=inferencer_args.max_new_tokens, 
                                    token_per_step=token_per_step, temperature=inferencer_args.temperature,
                                    end_string=end_string, input_dataset=input_dataset):
        delta = response[print_index:]
        seq = response
        print_index = len(response)

        yield delta, history + [(query, seq)]
        if flag_break:
            break

def predict(input, history=None): 
    if history is None:
        history = []
    for response, history in chat_stream(input, history):
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
            return f"Error parsing CIF file: {str(e)}"
    elif file_path.endswith('POSCAR'):
        try:
            structure = Poscar.from_file(file_path).structure
        except Exception as e:
            return f"Error parsing POSCAR file: {str(e)}"
    else:
        return "Unsupported file format"

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
        return f"Failed to open VESTA: {str(e)}"

    # 将结构转换为描述性字符串
    structure_description = structure_to_str(structure)
    return f"{structure_description}"

with gr.Blocks(css=css) as demo:
    gr.HTML(title)
    state = gr.State([])
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

    button.click(predict, [txt, state], [state] + text_boxes)

    gr.Markdown("# Text Converter for Crystal Structure")
    file_input = gr.File(label="Upload CIF or POSCAR file")
    output = gr.HTML(label="VESTA Visualization Status")

    def handle_file_upload(file):
        structure_description = visualize_structure(file)
        return structure_description, structure_description

    file_input.change(fn=handle_file_upload, inputs=file_input, outputs=txt)

demo.launch(share=True)
