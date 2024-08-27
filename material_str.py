from pyxtal import pyxtal
from p_tqdm import p_map
from sklearn.model_selection import train_test_split
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

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

def get_struct_from_str(structure_str):
  lines = structure_str.split('\n')
  abc = [float(x) for x in lines[2].split(':')[1].split()]
  angles = [float(x) for x in lines[3].split(':')[1].split()]
  lattice = Lattice.from_parameters(*abc, *angles)
  species = []
  coords = []
  for line in lines[8:]: 
      parts = line.split()
      species.append(parts[1])
      coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
  return Structure(lattice, species, coords)

def write_single_instance(struct, ene, train=True):
    """处理单个实例并返回格式化字符串"""
    # 假设 structure_to_str 是一个存在的函数，用于将结构转换为字符串
    from pyxtal import pyxtal
    try:
        symstr = structure_to_str(struct)
        if train:
            fine_tune_str = {'text': f'Input: Can this material structure be synthesized "{symstr}"? \n Output: {ene}'}
        else:
            fine_tune_str = {'input': f'Can this material structure be synthesized "{symstr}%"?', 'output': f'{ene}'}
    except:
        fine_tune_str = str(struct)
    return fine_tune_str

def write_cls_data_parallel(structs, energy,data_dict, train=True):
    """使用p_map并行处理数据"""
    # 使用p_map并行处理每个实例
    data_dict['instances'] = []
    instances = p_map(write_single_instance, structs, energy, [train] * len(structs))
    
    # 构建最终的数据字典
    data_dict['instances'] = instances
    return data_dict

# with open('/home/zhilong666/crystal_pre/alpaca/train/train_52002.json', 'r') as json_file:
#     data_ori = json.load(json_file)
# with open('/home/zhilong666/crystal_pre/alpaca/test/test_252.json', 'r') as json_file:
#     data_ori_test= json.load(json_file)

# structs_train,structs_test = train_test_split(structs_screen,test_size=0.1,random_state=42)
# energy_train,energy_test = train_test_split(if_sym,test_size=0.1,random_state=42)

# data_train = write_cls_data_parallel(structs_train, energy_train, data_ori,train=True)
# with open('train_sym.json', 'w') as json_file:
#     json.dump(data_train, json_file, ensure_ascii=False, indent=4)
# data_test = write_cls_data_parallel(structs_test, energy_test, data_ori_test, train=False)
# with open('test_sym.json', 'w') as json_file:
#     json.dump(data_test, json_file, ensure_ascii=False, indent=4)


