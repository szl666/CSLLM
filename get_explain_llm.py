import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import sys
import json

# sys.path.append("/home/aoboyang/local/captum")
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

def get_material_str(material_str):
    material_str_content = material_str['input'].split('"')[1]
    sp = material_str_content.split('|')[0].replace(' ','')
    lengths_and_angles = material_str_content.split('|')[1]
    lengths = ','.join(lengths_and_angles.split(',')[:3])
    angles = ','.join(lengths_and_angles.split(',')[3:])
    atoms = material_str_content.split('|')[2].split('->')
    atom_positions = []
    for atom in atoms:
        atom_symbol = atom.split('-')[0]
        atom_position = atom.split('-')[1]
        atom_positions.append(atom_symbol.replace('(','').replace(' ',''))
        atom_positions.append(atom_position.replace(')','').replace('%',''))
    base_str = "input: Can this material structure be synthesized \"{} |{}{}|"
    atom_str_key = int(len(atom_positions)/2)*'({}-{})->'
    atom_str = base_str+f" {atom_str_key}".strip('->')+'%\"?'
    material_info_str = [sp, lengths, angles] + atom_positions
    return material_info_str,atom_str

def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = "100000MB"

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=bnb_config,
    #     device_map="auto", # dispatch efficiently the model on the available ressources
    #     max_memory = {i: max_memory for i in range(n_gpus)}, 
    #     offload_folder='output_models/finetune_with_lora_sym/',
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)}, 
        offload_folder='output_models/finetune_with_lora_sym_50000',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        # load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

# model_name = "output_models/finetune_with_lora_sym/" 
model_name = "llama_hf/llama-7b-hf" 

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config)
model.eval()
# eval_prompt = "Dave lives in Palm Coast, FL and is a lawyer. His personal interests include"
# eval_prompt ="Can this material structure be synthesized \"225 |6.118,6.118,6.118,90.00,90.00,90.00| (Be-4b[0.5 0.5 0.5])->(In-4a[0. 0. 0.])->(Ru-8c[0.25 0.25 0.25])%\"?"

# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# with torch.no_grad():
#     output_ids = model.generate(model_input["input_ids"], max_new_tokens=15)[0]
#     response = tokenizer.decode(output_ids, skip_special_tokens=True)
#     print(response)
with open('/fs0/home/liqiang/LMFlow/data/sym_classify/test/test_sym1.json', 'r') as f:
    test_sym = json.load(f)

fa = FeatureAblation(model)
llm_attr = LLMAttribution(fa, tokenizer)
# inp = TextTokenInput(
#     eval_prompt, 
#     tokenizer,
#     skip_tokens=[1],  # skip the special token for the start of the text <s>
# )

# sv = ShapleyValues(model) 
# sv_llm_attr = LLMAttribution(sv, tokenizer)
# target = "playing guitar, hiking, and spending time with his family."

# attr_res = llm_attr.attribute(inp, target=target)
# print("attr to the output sequence:", attr_res.seq_attr.shape)  # shape(n_input_token)
# print("attr to the output tokens:", attr_res.token_attr.shape)  # shape(n_output_token, n_input_token)
# torch.save(attr_res,'attr_res.pt')
attr_res_all = []
for index,material_str in enumerate(test_sym['instances']):
    material_info_str, atom_str = get_material_str(material_str)
    inp = TextTemplateInput(
    template=atom_str, 
        values=material_info_str,
    )
    target = material_str['output']
    attr_res_all.append(llm_attr.attribute(inp, target=target, num_trials=3))
    if index%100 == 0:
        torch.save(attr_res_all,'attr_res_all.pt')

# inp = TextTemplateInput(
#     template="input: Can this material structure be synthesized \"{} |{}{}| ({}-{})->({}-{})->({}-{})%\"?", 
#     values=["225", "6.118,6.118,6.118","90.00,90.00,90.00", "Be","4b[0.5 0.5 0.5]", "In","4a[0. 0. 0.])", "Ru","8c[0.25 0.25 0.25]"],
#     baselines=["225", "7.326,7.326,7.326","90.00,90.00,90.00", "Gd","4a[0. 0. 0.])", "Mg","4b[0.5 0.5 0.5]", "Mg","8c[0.25 0.25 0.25]"],
# )
# # attr_res = llm_attr.attribute(inp, target=target)
# attr_res = sv_llm_attr.attribute(inp, target=target, num_trials=3)
# torch.save(attr_res,'attr_res7.pt')


# attr_res.plot_token_attr(show=True)