import sys
sys.path.append("../")
from rdkit import RDLogger, Chem
RDLogger.DisableLog('rdApp.*')
import torch
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from retro_star.common import prepare_starting_molecules, \
    plan_handle, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from models.graph2edits import Graph2Edits
import warnings
warnings.filterwarnings('ignore')
import argparse
import pandas as pd


def prepare_single_step_model(model_dump, device):
    logging.info('Loading trained graph2edits model from %s' % model_dump)
    checkpoint = torch.load(model_dump, map_location=device)
    config = checkpoint['saveables']
    single_step_model = Graph2Edits(**config, device=device)
    single_step_model.load_state_dict(checkpoint['state'])
    single_step_model.to(device)
    single_step_model.eval()
    return single_step_model

class RSPlanner:
    def __init__(self,
                 device='cuda',
                 expansion_topk=10,
                 iterations=500,
                 use_value_fn=True,
                 starting_molecules="../data/multi_step/retro_data/dataset/origin_dict.csv",
                 model_dump="../experiments/uspto_50k/epoch_123.pt",
                 save_folder="../data/multi_step/retro_data/saved_models",
                 value_model="best_epoch_final_4.pt",
                 fp_dim=2048,
                 max_routes_num = 10
                 ):
        self.device = device
        self.fp_dim = fp_dim

        setup_logger()
        device = torch.device(device)
        starting_mols = prepare_starting_molecules(starting_molecules)

        one_step = prepare_single_step_model(model_dump, device)

        self.model = ValueMLP(
            n_layers=1,
            fp_dim=fp_dim,
            latent_dim=128,
            dropout_rate=0.1
            ).to(device)
        model_f = '%s/%s' % (save_folder, value_model)
        logging.info('Loading value nn from %s' % model_f)
        self.model.load_state_dict(torch.load(model_f, map_location=device))
        self.model.eval()

        self.plan_handle = plan_handle(one_step, self, starting_mols, expansion_topk, iterations, max_routes_num)
        self.starting_mols = starting_mols

    def value_fn_run(self, mol):
        fp = smiles_to_fp(mol, fp_dim=self.fp_dim).reshape(1, -1)
        fp = torch.FloatTensor(fp).to(self.device)
        v = self.model(fp).item()
        return v

    def plan(self, target_mol, args, need_action=False):
        t0 = time.time()
        # (succ_num, msg, best_cost_route, best_len_route, topk_routes,
        #  best_diff_route_05, best_diff_route_03, best_diff_route_01) = self.plan_handle.molstar_run(target_mol)
        succ_num, msg, best_cost_route, best_len_route, topk_routes = self.plan_handle.molstar_run(target_mol, args)

        if succ_num > 0:
            best_diff_result = {
                'succ_num': succ_num,
                'time': time.time() - t0,
                'iter': msg[1],
                'routes': msg[0].serialize(need_action=need_action),
                'route_cost': msg[0].total_cost,
                'route_average_diff': msg[0].sum_diff/msg[0].length,
                'route_end_diff': msg[0].end_total_diff,
                'route_len': msg[0].length,
                'diffs': msg[0].diffs,
                'total_diffs': msg[0].total_diffs
            }
            best_cost_result = {
                'routes': best_cost_route.serialize(need_action=need_action),
                'route_cost': best_cost_route.total_cost,
                'route_average_diff': best_cost_route.sum_diff / best_cost_route.length,
                'route_end_diff': best_cost_route.end_total_diff,
                'route_len': best_cost_route.length
            }
            best_len_result = {
                'routes': best_len_route.serialize(need_action=need_action),
                'route_cost': best_len_route.total_cost,
                'route_average_diff': best_len_route.sum_diff / best_len_route.length,
                'route_end_diff': best_len_route.end_total_diff,
                'route_len': best_len_route.length
            }


            topk_results = []
            for route in topk_routes:
                result = {
                    'routes': route.serialize(need_action=need_action),
                    'route_cost': route.total_cost,
                    'route_average_diff': route.sum_diff/route.length,
                    'route_end_diff': route.end_total_diff,
                    'route_len': route.length
                }
                topk_results.append(result)

            return best_diff_result, best_cost_result, best_len_result, topk_results

        elif succ_num == -1:
            return None, None, None, 999

        else:
            logging.info('Synthesis path for %s not found. Please try increasing '
                         'the number of iterations.' % target_mol)
            return None, None, None, None


def main(args, planner):
    os.makedirs(args.save_dir, exist_ok=True)
    # read dataset
    if args.dataset != "" and args.smi == "":
        target_mol_df = pd.read_csv(args.dataset)
        target_mol_list = target_mol_df['SMILES'].to_list()
    elif args.smi != "" and args.dataset == "":
        target_mol_list = [args.smi]

    total = 0
    search_results = {'id':[],'target_mol': [], 'routes_num': [],
                      'best_diff_len': [], 'best_diff_cost': [], 'best_diff_end_diff': [], 'best_diff_aver_step_diff': [],
                      'best_len_len': [], 'best_len_cost': [], 'best_len_end_diff': [], 'best_len_aver_step_diff': [],
                      'best_cost_len':[], 'best_cost_cost': [], 'best_cost_end_diff': [], 'best_cost_aver_step_diff': [],
                      'best_diff_route': [], 'best_len_route': [], 'best_cost_route': []}

    for i, target_mol in enumerate(target_mol_list):

        print(f'search for {i} target_mol...')
        best_diff_result, best_cost_result, best_len_result, topk_results = planner.plan(target_mol, args, need_action=False)

        if topk_results is not None and topk_results != 999:
            total += 1
            search_results['id'].append(i)
            search_results['target_mol'].append(target_mol)
            search_results['routes_num'].append(best_diff_result['succ_num'])

            search_results['best_diff_len'].append(best_diff_result['route_len'])
            search_results['best_diff_cost'].append(best_diff_result['route_cost'])
            search_results['best_diff_end_diff'].append(best_diff_result['route_end_diff'])
            search_results['best_diff_aver_step_diff'].append(best_diff_result['route_average_diff'])

            search_results['best_len_len'].append(best_len_result['route_len'])
            search_results['best_len_cost'].append(best_len_result['route_cost'])
            search_results['best_len_end_diff'].append(best_len_result['route_end_diff'])
            search_results['best_len_aver_step_diff'].append(best_len_result['route_average_diff'])

            search_results['best_cost_len'].append(best_cost_result['route_len'])
            search_results['best_cost_cost'].append(best_cost_result['route_cost'])
            search_results['best_cost_end_diff'].append(best_cost_result['route_end_diff'])
            search_results['best_cost_aver_step_diff'].append(best_cost_result['route_average_diff'])

            search_results['best_diff_route'].append(best_diff_result['routes'])
            search_results['best_len_route'].append(best_len_result['routes'])
            search_results['best_cost_route'].append(best_cost_result['routes'])

        elif topk_results == 999:
            total += 1
            search_results['id'].append(i)
            search_results['target_mol'].append(target_mol)
            search_results['routes_num'].append(1)

            search_results['best_diff_len'].append(None)
            search_results['best_diff_cost'].append(None)
            search_results['best_diff_end_diff'].append(None)
            search_results['best_diff_aver_step_diff'].append(None)

            search_results['best_len_len'].append(None)
            search_results['best_len_cost'].append(None)
            search_results['best_len_end_diff'].append(None)
            search_results['best_len_aver_step_diff'].append(None)

            search_results['best_cost_len'].append(None)
            search_results['best_cost_cost'].append(None)
            search_results['best_cost_end_diff'].append(None)
            search_results['best_cost_aver_step_diff'].append(None)

            search_results['best_diff_route'].append(None)
            search_results['best_len_route'].append(None)
            search_results['best_cost_route'].append(None)

        else:
            search_results['id'].append(i)
            search_results['target_mol'].append(target_mol)
            search_results['routes_num'].append(0)

            search_results['best_diff_len'].append(None)
            search_results['best_diff_cost'].append(None)
            search_results['best_diff_end_diff'].append(None)
            search_results['best_diff_aver_step_diff'].append(None)

            search_results['best_len_len'].append(None)
            search_results['best_len_cost'].append(None)
            search_results['best_len_end_diff'].append(None)
            search_results['best_len_aver_step_diff'].append(None)

            search_results['best_cost_len'].append(None)
            search_results['best_cost_cost'].append(None)
            search_results['best_cost_end_diff'].append(None)
            search_results['best_cost_aver_step_diff'].append(None)

            search_results['best_diff_route'].append(None)
            search_results['best_len_route'].append(None)
            search_results['best_cost_route'].append(None)

        if (i+1) % args.save_every == 0:
            df = pd.DataFrame(search_results)
            df.to_csv(args.save_dir + "/" + args.save_name + '.csv')

    df = pd.DataFrame(search_results)
    df.to_csv(args.save_dir +  "/" + args.save_name + '.csv')
    print(f'Totally find {total}/{len(target_mol_list)} routes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph2Edits retrostar multi pre')
    parser.add_argument('--dataset', type=str, default='', help='pred data file path')
    parser.add_argument('--smi', type=str, default='', help='pred smiles')
    parser.add_argument('--save_dir', type=str, default='../retroscore_results', help='result file save dir')
    parser.add_argument('--save_name', type=str, default='multi_plan_results', help='result file save name(remove .csv)')
    parser.add_argument('--cost_weight', type=float, default=0.1, help='expand value compose cost weight')
    parser.add_argument('--filter_radio', type=float, default=0.3, help='filter radio as cost')
    parser.add_argument('--filter_coef', type=float, default=0.3, help='filter coef as diff')
    parser.add_argument('--save_every', type=int, default=50, help='every num save')
    args = parser.parse_args()

    planner = RSPlanner(
        device='cpu',
        use_value_fn=True,
        iterations=500,
        expansion_topk=10,
        model_dump='../experiments/uspto_full/epoch_65.pt',
        max_routes_num=10)

    main(args, planner)
