from pyxtal import pyxtal
from p_tqdm import p_map
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

def structure_to_str(structure):
    # 转换pymatgen结构为pyxtal结构
    px_structure = pyxtal()
    px_structure._from_pymatgen(structure)
    # 获取空间群信息
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
    """处理单个实例并返回格式化字符串。

    注意: 训练格式结尾是 "{symstr}"? , 测试格式是 "{symstr}%"? , 多一个 %。
    两者做结构级比对(判重、查训练/测试泄漏)前必须先统一去掉这个 %。
    """
    try:
        symstr = structure_to_str(struct)
    except Exception as err:
        # 转换失败的结构不能静默混进数据集, 返回 None 由上层过滤并计数
        print(f'[write_single_instance] 结构转换失败, 已跳过: {err}')
        return None
    if train:
        return {'text': f'Input: Can this material structure be synthesized "{symstr}"? \n Output: {ene}'}
    return {'input': f'Can this material structure be synthesized "{symstr}%"?', 'output': f'{ene}'}

def write_cls_data_parallel(structs, energy, data_dict, train=True):
    """使用p_map并行处理数据, 丢弃转换失败的结构并报出数量"""
    instances = p_map(write_single_instance, structs, energy, [train] * len(structs))
    valid = [instance for instance in instances if isinstance(instance, dict)]
    if len(valid) < len(instances):
        print(f'共 {len(instances)} 个结构, 转换失败并丢弃 {len(instances) - len(valid)} 个')
    data_dict['instances'] = valid
    return data_dict
