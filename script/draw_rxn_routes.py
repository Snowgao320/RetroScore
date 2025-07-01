import argparse
import os.path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from graphviz import Digraph, Graph


def mol_to_image(mol, n_id, size=(200, 200)):
    img = Draw.MolToImage(mol, size=size)
    save_path = os.path.join("..", "pred_results", "save_fig", f"{n_id}.png")
    img.save(save_path)
    return f'../pred_results/save_fig/{n_id}.png'  # 应设为绝对路径


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='../pred_results/one_mol_multi_all.csv', help='pred data file path')
    parser.add_argument('--col_name', type=str, default='route', help='result file save dir')
    parser.add_argument('--save_dir', type=str, default='../pred_results/one_mol_multi_fig', help='result file save dir')
    parser.add_argument('--top10', action='store_true', default=False, help='show top 10')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.read_csv(args.fpath)
    if args.top10:
        df = df.iloc[:10]
    routes_lst = df[args.col_name].to_list()
    for i, route in enumerate(routes_lst):
        # 第i条路线
        # 创建一个图
        G = Digraph(engine='dot',
         node_attr={'shape': 'box'}
         )
        id2name_dict = {}
        name2id_dict = {}
        n_id = -1
        for s, rxn_step in enumerate(route.split("|")):
            rxn_step = rxn_step.split(">")
            cp = rxn_step[0]
            cr = rxn_step[-1]
            if cp not in id2name_dict.values():
                mol = Chem.MolFromSmiles(cp)
                img = mol_to_image(mol, n_id+1+i)
                G.node(cp, label='', image=img, color='red')
                id2name_dict[n_id + 1] = cp
                name2id_dict[cp] = n_id + 1
                n_id += 1
            for cr_split in cr.split("."):
                if cr_split not in id2name_dict.values():
                    mol = Chem.MolFromSmiles(cr_split)
                    img = mol_to_image(mol, n_id+1+i)
                    G.node(cr_split, label='', image=img)
                    id2name_dict[n_id + 1] = cr_split
                    name2id_dict[cr_split] = n_id + 1
                    n_id += 1
                G.edge(cr_split, cp, weight="2.0")

        G.render(f'{args.save_dir}/{i+1}_graph', format='pdf', view=False)



        # # 将 SMILES 字符串转化为分子对象
        # molecule = Chem.MolFromSmiles(smi)
        # # 绘制分子结构图像
        # image = Draw.MolToImage(molecule)
        # image.save(os.path.join(args.save_dir, str(i+1) + '.png'))
