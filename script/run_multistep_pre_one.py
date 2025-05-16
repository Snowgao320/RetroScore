from rdkit import RDLogger, Chem
import sys
sys.path.append("../")
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
        succ_num, msg, best_cost_route, best_len_route, topk_routes = self.plan_handle.molstar_run(target_mol, args)

        if succ_num > 0:
            # 说明成功， 有路线
            best_diff_result = {
                'succ_num': succ_num,
                'time': time.time() - t0,
                'iter': msg[1],
                'routes': msg[0].serialize(need_action=need_action),
                'route_score': msg[0].total_score,
                'route_average_diff': msg[0].sum_diff/msg[0].length,
                'route_end_diff': msg[0].end_total_diff,
                'route_len': msg[0].length,
                'diffs': msg[0].diffs,
                'total_diffs': msg[0].total_diffs
            }
            best_cost_result = {
                'routes': best_cost_route.serialize(need_action=need_action),
                'route_score': best_cost_route.total_score,
                'route_average_diff': best_cost_route.sum_diff / best_cost_route.length,
                'route_end_diff': best_cost_route.end_total_diff,
                'route_len': best_cost_route.length
            }
            best_len_result = {
                'routes': best_len_route.serialize(need_action=need_action),
                'route_score': best_len_route.total_score,
                'route_average_diff': best_len_route.sum_diff / best_len_route.length,
                'route_end_diff': best_len_route.end_total_diff,
                'route_len': best_len_route.length
            }

            topk_results = []
            for route in topk_routes:
                result = {
                    'routes': route.serialize(need_action=need_action),
                    'route_score': route.total_score,
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
    # read dataset
    total = 0
    all_results = {'id':[],'target_mol': [], 'routes_num': [],'routes_id': [],
                      'len': [], 'end_diff': [], 'aver_step_diff': [], 'sum_confidence_score': [],
                     'route': []}

    search_results = {'id': [], 'target_mol': [], 'routes_num': [],
                      'best_diff_len': [], 'best_diff_score': [], 'best_diff_end_diff': [],
                      'best_diff_aver_step_diff': [],
                      'best_len_len': [], 'best_len_score': [], 'best_len_end_diff': [], 'best_len_aver_step_diff': [],
                      'best_sscore_len': [], 'best_sscore_score': [], 'best_sscore_end_diff': [],
                      'best_sscore_aver_step_diff': [],
                      'best_diff_route': [], 'best_len_route': [], 'best_sscore_route': []}
  
    target_mol = args.smi
    best_diff_result, best_cost_result, best_len_result, topk_results = planner.plan(target_mol, args, need_action=False)

    if topk_results is not None and topk_results != 999:
        total += 1
        search_results['id'].append(1)
        search_results['target_mol'].append(target_mol)
        search_results['routes_num'].append(best_diff_result['succ_num'])

        search_results['best_diff_len'].append(best_diff_result['route_len'])
        search_results['best_diff_score'].append(best_diff_result['route_score'])
        search_results['best_diff_end_diff'].append(best_diff_result['route_end_diff'])
        search_results['best_diff_aver_step_diff'].append(best_diff_result['route_average_diff'])

        search_results['best_len_len'].append(best_len_result['route_len'])
        search_results['best_len_score'].append(best_len_result['route_score'])
        search_results['best_len_end_diff'].append(best_len_result['route_end_diff'])
        search_results['best_len_aver_step_diff'].append(best_len_result['route_average_diff'])

        search_results['best_sscore_len'].append(best_cost_result['route_len'])
        search_results['best_sscore_score'].append(best_cost_result['route_score'])
        search_results['best_sscore_end_diff'].append(best_cost_result['route_end_diff'])
        search_results['best_sscore_aver_step_diff'].append(best_cost_result['route_average_diff'])

        search_results['best_diff_route'].append(best_diff_result['routes'])
        search_results['best_len_route'].append(best_len_result['routes'])
        search_results['best_sscore_route'].append(best_cost_result['routes'])

        # for all results
        for ir, r in enumerate(topk_results):
            # for all routes
            all_results['id'].append(1)
            all_results['target_mol'].append(target_mol)
            all_results['routes_num'].append(best_diff_result['succ_num'])
            all_results['routes_id'].append(ir+1)
            all_results['len'].append(r['route_len'])
            all_results['end_diff'].append(r['route_end_diff'])
            all_results['aver_step_diff'].append(r['route_average_diff'])
            all_results['sum_confidence_score'].append(r['route_score'])
            all_results['route'].append(r['routes'])

    elif topk_results == 999:
        total += 1
        search_results['id'].append(1)
        search_results['target_mol'].append(target_mol)
        search_results['routes_num'].append(1)

        search_results['best_diff_len'].append(None)
        search_results['best_diff_score'].append(None)
        search_results['best_diff_end_diff'].append(None)
        search_results['best_diff_aver_step_diff'].append(None)

        search_results['best_len_len'].append(None)
        search_results['best_len_score'].append(None)
        search_results['best_len_end_diff'].append(None)
        search_results['best_len_aver_step_diff'].append(None)

        search_results['best_sscore_len'].append(None)
        search_results['best_sscore_cost'].append(None)
        search_results['best_sscore_end_diff'].append(None)
        search_results['best_sscore_aver_step_diff'].append(None)

        search_results['best_diff_route'].append(None)
        search_results['best_len_route'].append(None)
        search_results['best_sscore_route'].append(None)

        # for all results
        all_results['id'].append(1)
        all_results['target_mol'].append(target_mol)
        all_results['routes_num'].append(1)
        all_results['routes_id'].append("target in stock")
        all_results['len'].append(None)
        all_results['end_diff'].append(None)
        all_results['aver_step_diff'].append(None)
        all_results['sum_confidence_score'].append(None)
        all_results['route'].append("target in stock")

    else:
        search_results['id'].append(1)
        search_results['target_mol'].append(target_mol)
        search_results['routes_num'].append(0)

        search_results['best_diff_len'].append(None)
        search_results['best_diff_score'].append(None)
        search_results['best_diff_end_diff'].append(None)
        search_results['best_diff_aver_step_diff'].append(None)

        search_results['best_len_len'].append(None)
        search_results['best_len_score'].append(None)
        search_results['best_len_end_diff'].append(None)
        search_results['best_len_aver_step_diff'].append(None)

        search_results['best_sscore_len'].append(None)
        search_results['best_sscore_cost'].append(None)
        search_results['best_sscore_end_diff'].append(None)
        search_results['best_sscore_aver_step_diff'].append(None)

        search_results['best_diff_route'].append(None)
        search_results['best_len_route'].append(None)
        search_results['best_sscore_route'].append(None)

        # for all results
        all_results['id'].append(1)
        all_results['target_mol'].append(target_mol)
        all_results['routes_num'].append(0)
        all_results['routes_id'].append("no route")
        all_results['len'].append(None)
        all_results['end_diff'].append(None)
        all_results['aver_step_diff'].append(None)
        all_results['sum_confidence_score'].append(None)
        all_results['route'].append("no route")

    recomd_df = pd.DataFrame(search_results)
    recomd_df.to_csv(args.save_dir + "/"+ args.save_name + '.csv')
    all_df = pd.DataFrame(all_results)
    all_df.to_csv(args.save_dir + "/"+ args.save_name + '_all.csv')
    # print(f'Totally find {total}/{len(target_mol_list)} routes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph2Edits retrostar multi pre')
    parser.add_argument('--smi', type=str, default='CON(C)C(=O)CC1COCCN1C(=O)OC(C)(C)C', help='pred data smiles')
    parser.add_argument('--save_dir', type=str, default='../', help='result file save dir')
    parser.add_argument('--save_name', type=str, default='test1', help='result file save name(remove .csv)')
    parser.add_argument('--cost_weight', type=float, default=0.1, help='expand value compose cost weight')
    parser.add_argument('--filter_radio', type=float, default=0.3, help='filter radio as cost')
    parser.add_argument('--filter_coef', type=float, default=0.3, help='filter coef as diff')
    parser.add_argument('--routes_num', type=int, default=10, help='max_routes_num')
    parser.add_argument('--iter', type=int, default=500, help='iterations num')
    parser.add_argument('--save_every', type=int, default=50, help='every num save')
    args = parser.parse_args()

    planner = RSPlanner(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_value_fn=True,
        iterations=args.iter,
        expansion_topk=10,
        model_dump='../experiments/uspto_full/epoch_65.pt',
        max_routes_num=args.routes_num,)


    main(args, planner)




