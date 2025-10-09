import argparse
import os.path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from graphviz import Digraph
import pickle
import shutil
import sys
sys.path.append("../")


def draw_route(route: str, tsf_dir: str):
    G = Digraph(engine='dot',
                node_attr={'shape': 'box'}
                )
    id2name_dict = {}
    name2id_dict = {}
    node_id = 0
    for step, reaction in enumerate(route.split("|")):
        p = reaction.split(">")[0]
        r = reaction.split(">")[-1]
        if p not in id2name_dict.values():
            p_mol = Chem.MolFromSmiles(p)
            p_img = mol_to_image(p_mol, node_id, tsf_dir)
            G.node(
                p, label=p, labelloc="t", labeljust="c", image=p_img, color='red')
            id2name_dict[node_id] = p
            name2id_dict[p] = node_id
            node_id += 1
        # add reactants
        for _r_smi in r.split("."):
            _r_mol = Chem.MolFromSmiles(_r_smi)
            _r_img = mol_to_image(_r_mol, node_id, tsf_dir)
            G.node(
                _r_smi, label=_r_smi, labelloc="t", labeljust="c", image=_r_img)
            id2name_dict[node_id] = _r_smi
            name2id_dict[_r_smi] = node_id
            G.edge(_r_smi, p, weight="2.0")
            node_id += 1
    return G


def mol_to_image(mol, node_id, tsf_dir, size=(200, 200)):
    image = Draw.MolToImage(mol, size=size)
    image_fpath = os.path.join(tsf_dir, f"{node_id}.png")
    image.save(image_fpath)
    return os.path.abspath(image_fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='../pred_results/routes_pred_all_routes.pkl',
                        help='Path to the pred file, including multi-step routes')
    parser.add_argument('--save_dir', type=str, default='../pred_results/draw',
                        help='result file save dir')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    # 判断给定的文件是csv还是pkl; csv -> 画3个推荐路线；pkl -> 画所有路线
    if args.fpath[-3:] == 'pkl':
        all_routes_results = pickle.load(open(args.fpath, 'rb'))
        target_lst = []
        for target_id, (target_smi, total_routes) in enumerate(all_routes_results):
            # 先创建保存和中转子图片使用的文件夹
            routes_img_path = args.save_dir + f"/target_{target_id}/total_routes_set"
            os.makedirs(routes_img_path, exist_ok=True)
            transfer_path = args.save_dir + f"/target_{target_id}/draw_transfer_station"
            os.makedirs(transfer_path, exist_ok=True)

            for route_id, r in enumerate(total_routes):
                r_draw = draw_route(r.serialize(), transfer_path)
                r_draw.render(f"route_{route_id}", directory=routes_img_path, format='pdf', view=False)

            # 该化合物的路线全部绘制完成，保存单独一个文件夹
            target_lst.append((target_id, target_smi))
            shutil.rmtree(transfer_path)    # 删除中转使用的文件夹

        # 构成id -> target smiles的对照表
        target_id, target_smi = zip(*target_lst)
        df = pd.DataFrame({
            "target_id": target_id,
            "target_smi": target_smi
        })
        df.to_csv(args.save_dir+"/id2target_total_routes.csv", index=False)

    elif args.fpath[-3:] == 'csv':
        df = pd.read_csv(args.fpath)
        size = len(df)
        target_lst = []
        for target_id in range(size):
            target_smi = df.iloc[target_id]['target_mol']
            length_best = df.iloc[target_id]['len_best_route']
            score_best = df.iloc[target_id]['score_best_route']
            multi_stage_best = df.iloc[target_id]['multi_stage_best_route']

            # 先创建保存和中转子图片使用的文件夹
            routes_img_path = args.save_dir + f"/target_{target_id}/recommend_routes"
            os.makedirs(routes_img_path, exist_ok=True)
            transfer_path = args.save_dir + f"/target_{target_id}/draw_transfer_station"
            os.makedirs(transfer_path, exist_ok=True)

            length_best_draw = draw_route(length_best, transfer_path)
            length_best_draw.render(f'{routes_img_path}/route_length_best', format='pdf', view=False)
            score_best_draw = draw_route(score_best, transfer_path)
            score_best_draw.render(f'{routes_img_path}/route_sum_score_best', format='pdf', view=False)
            multi_stage_best_draw = draw_route(multi_stage_best, transfer_path)
            multi_stage_best_draw.render(f'{routes_img_path}/route_multi_stage_best', format='pdf', view=False)

            # 该化合物的路线全部绘制完成，保存单独一个文件夹
            target_lst.append((target_id, target_smi))
            shutil.rmtree(transfer_path)  # 删除中转使用的文件夹

        # 构成id -> target smiles的对照表
        target_id, target_smi = zip(*target_lst)
        df = pd.DataFrame({
            "target_id": target_id,
            "target_smi": target_smi
        })
        df.to_csv(args.save_dir + "/id2target_recommend.csv", index=False)


