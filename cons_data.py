import os
import re
# import lmdb
import json
import pickle
import numpy as np
import pandas as pd
from p_tqdm import p_map
from pyxtal.symmetry import Group
from sklearn.model_selection import train_test_split
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ocpmodels.datasets import SinglePointLmdbDataset
from collections import defaultdict
from pymatgen.core import Structure
from pyxtal import pyxtal
from pyxtal.molecular_crystal import molecular_crystal
from pyxtal.lattice import Lattice
import matplotlib.cm as cm 
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib.pyplot import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib.patches as patches


oqmd_structs = SinglePointLmdbDataset({"src": "oqmd_structs.lmdb"})
mp_structs = SinglePointLmdbDataset({"src": "mp_structs.lmdb"})
carolina_matdb_structs = SinglePointLmdbDataset({"src": "carolina_matdb_structs.lmdb"})
jarvis_structs = SinglePointLmdbDataset({"src": "jarvis_structs.lmdb"})
icsd_structs = SinglePointLmdbDataset({"src": "icsd_structs.lmdb"})
structs_all = oqmd_structs+mp_structs+carolina_matdb_structs+jarvis_structs
structs_all = [Structure.from_dict(struct) for struct in structs_all]

screened_structs = []
for index in range(len(icsd_structs)):
    if index%1000==0:
        print(index)
# for index in range(500):
    struct = Structure.from_dict(icsd_structs[index])
    if struct.is_ordered:
        if len(struct)<=40 and len(struct.composition)<=7 and 'D' not in struct.composition.element_composition and 'T' not in struct.composition.element_composition and max(struct.atomic_numbers)<=92:
            screened_structs.append(struct)
train, val_test = train_test_split(list(range(len(screened_structs))), test_size=0.2, random_state=42)
val, test = train_test_split(val_test, test_size=0.5, random_state=42)
cif_strs_train = [];materials_ids_train = []
cif_strs_val = [];materials_ids_val = []
cif_strs_test = [];materials_ids_test = []
for train_index in train:
    if screened_structs[train_index].is_ordered:
        cif_strs_train.append(screened_structs[train_index].to(fmt="cif"))
        materials_ids_train.append(screened_structs[train_index].formula.replace(' ',''))
for val_index in val:
    if screened_structs[val_index].is_ordered:
        cif_strs_val.append(screened_structs[val_index].to(fmt="cif"))
        materials_ids_val.append(screened_structs[val_index].formula.replace(' ',''))
for test_index in test:
    if screened_structs[test_index].is_ordered:
        cif_strs_test.append(screened_structs[test_index].to(fmt="cif"))
        materials_ids_test.append(screened_structs[test_index].formula.replace(' ',''))

def get_valid_cif(cif_strs,materials_ids):
    cif_strs_new = [];materials_ids_new = []
    for cif_str,materials_id in zip(cif_strs,materials_ids):
        try:
            crystal = Structure.from_str(cif_str, fmt='cif')
            if crystal:
                cif_strs_new.append(cif_str)
                materials_ids_new.append(materials_id)
        except:
            pass
    return cif_strs_new,materials_ids_new

cif_strs_train,materials_ids_train = get_valid_cif(cif_strs_train,materials_ids_train)
cif_strs_val,materials_ids_val = get_valid_cif(cif_strs_val,materials_ids_val)
cif_strs_test,materials_ids_test = get_valid_cif(cif_strs_test,materials_ids_test)
train_df = pd.DataFrame({'index':train[:len(cif_strs_train)], 'material_id':materials_ids_train, 'cif':cif_strs_train, 'pbes_gap':[0 for i in range(len(cif_strs_train))]})
val_df = pd.DataFrame({'index':val[:len(cif_strs_val)], 'material_id':materials_ids_val, 'cif':cif_strs_val, 'pbes_gap':[0 for i in range(len(cif_strs_val))]})
test_df = pd.DataFrame({'index':test[:len(cif_strs_test)], 'material_id':materials_ids_test, 'cif':cif_strs_test, 'pbes_gap':[0 for i in range(len(cif_strs_test))]})
train_df.to_csv('/home/zhilong666/sym_structs/train.csv', index=False)
val_df.to_csv('/home/zhilong666/sym_structs/val.csv', index=False)
test_df.to_csv('/home/zhilong666/sym_structs/test.csv', index=False)


def write_single_instance(formula, ene, train=True):
    # from pyxtal import pyxtal
    """处理单个实例并返回格式化字符串"""
    # 假设 structure_to_str 是一个存在的函数，用于将结构转换为字符串
    ene = str(ene)
    if train:
        fine_tune_str = {'text': f'Input: How can this material structure be synthesized "{formula}"? \n Output: {ene}'}
    else:
        fine_tune_str = {'input': f'How can this material structure be synthesized "{formula}%"?', 'output': f'{ene}'}
    return fine_tune_str

def write_cls_data_parallel(structs, energy,data_dict, train=True):
    """使用p_map并行处理数据"""
    # 使用p_map并行处理每个实例
    data_dict['instances'] = []
    instances = p_map(write_single_instance, structs, energy, [train] * len(structs))
    instances = [instance for instance in instances if type(instance) == dict]
    # 构建最终的数据字典
    data_dict['instances'] = instances
    return data_dict

def get_data_formula(llm_info):
    formulas = list(llm_info.keys())
    # formulas = [i['formula'] for i in llm_info.values()]
    llm_info_all = [i for i in llm_info.values()]
    llm_method = [i['method'] for i in llm_info.values()]
    llm_precursors = [i['precursors'] for i in llm_info.values()]
    llm_operations = [i['operations'] for i in llm_info.values()]
    return formulas,llm_info_all, llm_method,llm_precursors,llm_operations

def write_llm_json(structs_train, structs_test, energy_train,energy_test,json_name,folder):
    data_train = write_cls_data_parallel(structs_train, energy_train, data_ori,train=True)
    with open(f'{folder}/train/train_{json_name}.json', 'w') as json_file:
        json.dump(data_train, json_file, ensure_ascii=False, indent=4)
    data_test = write_cls_data_parallel(structs_test, energy_test, data_ori_test, train=False)
    with open(f'{folder}/test/test_{json_name}.json', 'w') as json_file:
        json.dump(data_test, json_file, ensure_ascii=False, indent=4)