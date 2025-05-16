import os

from rdkit import Chem
from rdkit.Chem import Draw
import argparse
import pandas as pd



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str, default='../pred_results/batch_mol_single.csv', help='pred data file path')
    parser.add_argument('--save_dir', type=str, default='../pred_results/one_batch_mol_single', help='result file save dir')
    parser.add_argument('--top10', action='store_true', default=False, help='show top 10')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.read_csv(args.fpath)
    if args.top10:
        df = df.iloc[:10]
    try:
        smi_lst = df['reactants'].to_list()
    except:
        smi_lst = df['target_mol'].to_list()
    for i, smi in enumerate(smi_lst):
        # 将 SMILES 字符串转化为分子对象
        molecule = Chem.MolFromSmiles(smi)
        # 绘制分子结构图像
        image = Draw.MolToImage(molecule)
        image.save(os.path.join(args.save_dir, str(i+1) + '.png'))