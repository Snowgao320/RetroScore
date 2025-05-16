import sys

import numpy as np
import pandas as pd
import os
import argparse
import copy
import torch
sys.path.append("../")
from rdkit import Chem, RDLogger
from models import Graph2Edits, BeamSearch
lg = RDLogger.logger()
lg.setLevel(4)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def canonicalize(mol):
    try:
        tmp = Chem.RemoveHs(mol)
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return tmp
    except:
        return None


def canonicalize_prod(p):
    p = copy.deepcopy(p)
    p_mol = Chem.MolFromSmiles(p)
    if p_mol is None:
        return None
    p_mol = canonicalize(p_mol)
    if p_mol is None:
        return None

    for atom in p_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    p = Chem.MolToSmiles(p_mol)
    return p



def main(args):
    # 加载模型
    checkpoint = torch.load(os.path.join("../experiments/uspto_full", 'epoch_65.pt'), map_location=torch.device(DEVICE))
    config = checkpoint['saveables']

    model = Graph2Edits(**config, device=DEVICE)
    model.load_state_dict(checkpoint['state'])
    model.to(DEVICE)
    model.eval()

    top_k = np.zeros(args.beam_size)
    beam_model = BeamSearch(model=model, step_beam_size=10,
                            beam_size=args.beam_size, use_rxn_class=False)


    if args.smi != '' and args.fpath == '':
        # 只预测一个分子
        p = canonicalize_prod(args.smi)

        with torch.no_grad():
            top_k_results = beam_model.run(
                prod_smi=p, max_steps=args.max_steps, rxn_class=None)

        sum_topk_results = {
            "id": [],
            "target": [],
            "reactants": [],
            "scores": [],
            'edits': []
        }

        # {'reactants': reactants,
        #         'scores': scores,
        #         'edits': reaction_edits}
        # 将结果写入文件
        for i in range(0, len(top_k_results['reactants'])):
            sum_topk_results['id'].append(i+1)
            sum_topk_results['target'].append(args.smi)
            sum_topk_results['reactants'].append(top_k_results['reactants'][i])
            sum_topk_results['scores'].append(top_k_results['scores'][i])
            sum_topk_results['edits'].append(top_k_results['edits'][i])

        df = pd.DataFrame(sum_topk_results)
        df.to_csv(f'{args.save_dir}/{args.save_name}.csv', index=False)

    elif args.smi == '' and args.fpath != '':
        df = pd.read_csv(args.fpath)
        target_lst = df['SMILES'].to_list()

        sum_topk_results = {
            "id": [],
            "target": [],
            "reactants": [],
            "scores": [],
            'edits': []
        }
        for i, smi in enumerate(target_lst):
            p = canonicalize_prod(smi)

            with torch.no_grad():
                top_k_results = beam_model.run(
                    prod_smi=p, max_steps=args.max_steps, rxn_class=None)

            id_col = [i+1 for ii in range(1, len(top_k_results['reactants']) + 1)]
            sum_topk_results['id'] += id_col
            sum_topk_results['target'] += [smi]*len(top_k_results['reactants'])
            sum_topk_results['reactants'] += top_k_results['reactants']
            sum_topk_results['scores'] += top_k_results['scores']
            sum_topk_results['edits'] += top_k_results['edits']

        df = pd.DataFrame(sum_topk_results)
        df.to_csv(f'{args.save_dir}/{args.save_name}.csv', index=False)

    else:
        raise ValueError("Please input valid smi or upload normative file!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph2Edits single step pre')
    parser.add_argument('--smi', type=str, default='', help='pred smiles')
    parser.add_argument('--fpath', type=str, default='', help='pred data file path')
    parser.add_argument('--save_dir', type=str, default='../pred_results', help='result file save dir')
    parser.add_argument('--save_name', type=str, default='test1', help='result file save name(remove .csv)')
    parser.add_argument('--beam_size', type=int,
                        default=10, help='Beam search width')
    parser.add_argument('--max_steps', type=int, default=9,
                        help='maximum number of edit steps')
    args = parser.parse_args()
    main(args)
