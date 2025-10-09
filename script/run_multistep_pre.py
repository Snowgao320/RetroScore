import sys

import torch.cuda

sys.path.append("../")
from rdkit import RDLogger, Chem
import logging
import pickle
import os
import warnings
import argparse
import pandas as pd
from retro_star.common import RSPlanner
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    # read dataset
    if args.dataset != "" and args.smi == "":
        if args.dataset.split(".")[-1] == "csv" or args.dataset.split(".")[-1] == "txt":
            target_mol_df = pd.read_csv(args.dataset)
            target_mol_list = target_mol_df['SMILES'].to_list()
        elif args.dataset.split(".")[-1] == "pkl":
            routes = pickle.load(open(args.dataset, 'rb'))
            logging.info('%d routes extracted from %s loaded' % (len(routes), 'routes_possible_test_hard.pkl'))
            target_mol_list = [route[0].split('>')[0] for route in routes]
        else:
            raise ValueError('dataset must be either csv or pkl')

    elif args.smi != "" and args.dataset == "":
        target_mol_list = [args.smi]

    succ_total = 0
    total_res_lst = []
    if args.mode != "find_param":
        total_routes_lst = []
        search_results = {
            'target_mol': [],
            'routes_num': [],
            'time_cost': [],
            'first_time_cost': [],
            'iter': [],

            'score_best_route': [],
            'score_best_sum_score': [],
            'score_best_length': [],
            'score_best_sum_step_dists': [],
            'score_best_end_dist': [],

            'len_best_route': [],
            'len_best_sum_score': [],
            'len_best_length': [],
            'len_best_sum_step_dists': [],
            'len_best_end_dist': [],

            'multi_stage_best_route': [],
            'multi_stage_best_sum_score': [],
            'multi_stage_best_length': [],
            'multi_stage_best_sum_step_dists': [],
            'multi_stage_best_end_dist': [],
            }
    # 按顺序逐一搜索
    for i, target_mol in enumerate(target_mol_list):
        print(f'search for {i+1} target_mol...')
        succ_num, result = planner.plan(target_mol, args, need_action=False)

        if succ_num > 0 or succ_num == -1:
            succ_total += 1

        total_res_lst.append(result)
        if args.mode != "find_param":
            search_results['target_mol'].append(target_mol)
            for k, v in result.items():
                if k == "total_routes":
                    total_routes_lst.append((target_mol, result['total_routes']))
                else:
                    search_results[k].append(result[k])

        if (i+1) % args.save_every == 0:
            with open(os.path.join(args.save_dir, args.save_name + ".pkl"), 'wb') as f:
                pickle.dump(total_res_lst, f)

            if args.mode != "find_param":
                df = pd.DataFrame(search_results)
                df.to_csv(os.path.join(args.save_dir, args.save_name + ".csv"), index=False)
                with open(os.path.join(args.save_dir, args.save_name + "_all_routes.pkl"), 'wb') as f:
                    pickle.dump(total_routes_lst, f)
    # final save
    with open(os.path.join(args.save_dir, args.save_name + ".pkl"), 'wb') as f:
        pickle.dump(total_res_lst, f)

    if args.mode != "find_param":
        df = pd.DataFrame(search_results)
        df.to_csv(os.path.join(args.save_dir, args.save_name + ".csv"), index=False)
        with open(os.path.join(args.save_dir, args.save_name + "_all_routes.pkl"), 'wb') as f:
            pickle.dump(total_routes_lst, f)

    print(f'Totally find {succ_total}/{len(target_mol_list)} routes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph2Edits-retro*d multi-step prediction')
    parser.add_argument('--dataset', type=str, default='',
                        help='pred data file path')
    parser.add_argument('--smi', type=str, default='', help='pred smiles')
    parser.add_argument('--save_dir', type=str, default='../pred_results',
                        help='result file save dir')
    parser.add_argument('--save_name', type=str, default='routes_pred',
                        help='result file save name(remove .csv)')
    parser.add_argument('--cost_weight', type=float, default=0.1,
                        help='The weight of confidence score for expanding function')
    parser.add_argument('--filter_ratio', type=float, default=0.9,
                        help='filter ratio with average confidence score of multi-stage screening')
    parser.add_argument('--coef', type=float, default=0.3,
                        help='Coef with end dist of multi-stage screening')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save interval')
    parser.add_argument('--mode', type=str, default="sure_param", choices=['find_param', 'sure_param'],
                        help='Prediction mode')
    parser.add_argument('--print_interval', type=int, default=100,
                        help='Print interval for iteration')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to use')
    parser.add_argument('--iterations', type=int, default=500, help='Max iterations')
    parser.add_argument('--max_routes', type=int, default=10, help='Max number of routes to search')
    parser.add_argument('--expansion_topk', type=int, default=10,
                        help='Expansion topk candidates for one step')
    parser.add_argument('--stock_path', type=str, default="../data/multi_step/retro_data/dataset/origin_dict.csv",
                        help='The final compounds stock file path')
    parser.add_argument('--one_step_model', type=str, default="../experiments/uspto_full/epoch_65.pt",
                        help='One step model file path')
    parser.add_argument('--value_model', type=str, default="../data/multi_step/retro_data/saved_models/best_epoch_final_4.pt",
                        help='Value model file path')
    args = parser.parse_args()

    planner = RSPlanner(
        device=args.device,
        expansion_topk=args.expansion_topk,
        iterations=args.iterations,
        starting_molecules=args.stock_path,
        model_dump=args.one_step_model,
        value_model=args.value_model,
        fp_dim=2048,
        max_routes_num=args.max_routes)

    main()
