import pickle
import pandas as pd
import logging
import torch
from models.beam_search import BeamSearch
from retro_star.alg import molstar
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from models.graph2edits import Graph2Edits
from rxnmapper import RXNMapper


class PlanHandle:
    def __init__(self, one_step, value_fn, fp_dim, starting_mols, expansion_topk, iterations, max_routes_num, device):
        self.beam_model = BeamSearch(model=one_step, step_beam_size=expansion_topk,
                            beam_size=expansion_topk, use_rxn_class=False)
        self.value_fn = value_fn
        self.starting_mols = starting_mols
        self.iterations = iterations
        self.max_routes_num = max_routes_num
        self.fp_dim = fp_dim
        self.device = device

    def beam_model_run(self, x):
        with torch.no_grad():
            return self.beam_model.run(prod_smi=x, max_steps=9, rxn_class=None)

    def molstar_run(self, x, args):
        m = molstar(
            target_mol=x,
            starting_mols=self.starting_mols,
            expand_fn=self,
            value_fn=self.value_fn,
            iterations=self.iterations,
            max_routes_num=self.max_routes_num,
            mapper=RXNMapper(),
            args=args
        )
        return m


def prepare_starting_molecules(filename):
    logging.info('Loading starting molecules from %s' % filename)

    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols


def prepare_single_step_model(model_dump, device):
    logging.info('Loading trained graph2edits model from %s' % model_dump)
    checkpoint = torch.load(model_dump, map_location=device, weights_only=False)
    config = checkpoint['saveables']
    single_step_model = Graph2Edits(**config, device=device)
    single_step_model.load_state_dict(checkpoint['state'])
    single_step_model.to(device)
    single_step_model.eval()

    return single_step_model


class RSPlanner:
    def __init__(self,
                 device = 'cuda',
                 expansion_topk = 10,
                 iterations = 500,
                 starting_molecules = "../data/multi_step/retro_data/stocks/origin_dict.csv",
                 model_dump = "../experiments/uspto_full/epoch_65.pt",
                 value_model = "../data/multi_step/retro_data/saved_models/best_epoch_final_4.pt",
                 fp_dim = 2048,
                 max_routes_num = 10
                 ):

        setup_logger()
        device = torch.device(device)
        starting_mols = prepare_starting_molecules(starting_molecules)
        one_step = prepare_single_step_model(model_dump, device)

        self.model = ValueMLP(
            n_layers=1,
            fp_dim=fp_dim,
            latent_dim=128,
            dropout_rate=0.1,
            device=device
            ).to(device)
        logging.info('Loading value nn from %s' % value_model)
        self.model.load_state_dict(torch.load(value_model, map_location=device))
        self.model.eval()

        self.plan_handle = PlanHandle(
            one_step, self.model, fp_dim, starting_mols, expansion_topk, iterations, max_routes_num, device
        )

    def plan(self, target_mol, args, need_action=False):
        res = self.plan_handle.molstar_run(target_mol, args)
        succ_num, (time_cost, time_cost_f), iteration, total_routes, score_best, len_best, multi_process_best_set = res

        if succ_num == 0:
            logging.info('Synthesis path for %s not found. Please try increasing '
                         'the number of iterations.' % target_mol)

        if succ_num > 0 and args.mode == "find_param":  # 有路线且要确定参数组合-> 保存pkl版本，用于后续作图等
            best_result = {
                'routes_num': succ_num,
                'time_cost': time_cost,
                'first_time_cost': time_cost_f,
                'iter': iteration,
                'score_best_route': score_best,
                'len_best_route': len_best,
                'multi_process_routes': multi_process_best_set,   # {param_combin: best_route}
                'total_routes': total_routes
            }
        elif succ_num > 0 and args.mode != "find_param":  # 有路线且指定参数组合-> 保存csv版本
            multi_process_best = list(multi_process_best_set.values())[0]
            best_result = {
                'routes_num': succ_num,
                'time_cost': time_cost,
                'first_time_cost': time_cost_f,
                'iter': iteration,

                'score_best_route': score_best.serialize(need_action=need_action),
                'score_best_sum_score': score_best.sum_scores,
                'score_best_length': score_best.length,
                'score_best_sum_step_dists': score_best.sum_step_dists,
                'score_best_end_dist': score_best.end_dist,

                'len_best_route': len_best.serialize(need_action=need_action),
                'len_best_sum_score': len_best.sum_scores,
                'len_best_length': len_best.length,
                'len_best_sum_step_dists': len_best.sum_step_dists,
                'len_best_end_dist': len_best.end_dist,

                'multi_stage_best_route': multi_process_best.serialize(need_action=need_action),
                'multi_stage_best_sum_score': multi_process_best.sum_scores,
                'multi_stage_best_length': multi_process_best.length,
                'multi_stage_best_sum_step_dists': multi_process_best.sum_step_dists,
                'multi_stage_best_end_dist': multi_process_best.end_dist,

                'total_routes': total_routes
            }
        elif succ_num <= 0:  # 不在库且没有找到路线; 分子在库-> 保存csv版本
            best_result = {
                'routes_num': succ_num,
                'time_cost': time_cost,
                'first_time_cost': time_cost_f,
                'iter': iteration,

                'score_best_route': None,
                'score_best_sum_score': None,
                'score_best_length': None,
                'score_best_sum_step_dists': None,
                'score_best_end_dist': None,

                'len_best_route': None,
                'len_best_sum_score': None,
                'len_best_length': None,
                'len_best_sum_step_dists': None,
                'len_best_end_dist': None,

                'multi_stage_best_route': None,
                'multi_stage_best_sum_score': None,
                'multi_stage_best_length': None,
                'multi_stage_best_sum_step_dists': None,
                'multi_stage_best_end_dist': None,

                'total_routes': None
            }
        return succ_num, best_result

