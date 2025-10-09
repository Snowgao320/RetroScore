import copy
import time
from itertools import product
import numpy as np
import logging
from rdkit import Chem
from retro_star.alg.mol_tree import MolTree
from utils.calculate_edit_distance import (map_step_rxn, calculate_edits_distance,
                                           remap_smi_according_to_infer, normalize_edit_dist)


def canonicalize(mol):
    mol = Chem.RemoveHs(mol)
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))
    return mol


def canonicalize_prod(p):
    p_mol = Chem.MolFromSmiles(p)
    if p_mol is None:
        return None
    p_mol = canonicalize(p_mol)
    for atom in p_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    p = Chem.MolToSmiles(p_mol, isomericSmiles=True)
    return p


def molstar(target_mol, starting_mols, expand_fn, value_fn, iterations, max_routes_num, mapper, args):
    try:
        mol_tree = MolTree(
            N=max_routes_num,
            target_mol=target_mol,
            known_mols=starting_mols,
            value_fn=value_fn
        )

        i = 0
        start_time = time.time()
        time_cost_f = None
        if not mol_tree.succ:   # 目标产物不在库；需要搜索路线
            for i in range(1, iterations + 1):
                if i % args.print_interval == 0:
                    print(f'Run for {i} iters!')

                metric = []
                for m in mol_tree.mol_nodes:
                    if m.open:
                        metric.append(m.v_target())
                    else:
                        metric.append(np.inf)
                metric = np.array(metric)

                if np.min(metric) == np.inf:
                    logging.info('No open nodes!')
                    break

                m_next = mol_tree.mol_nodes[np.argmin(metric)]
                assert m_next.open

                cur_p_mol = canonicalize_prod(m_next.mol)       # single pred model forward
                if cur_p_mol is not None:
                    try:
                        result = expand_fn.beam_model_run(cur_p_mol)
                    except Exception as e:
                        result = {'reactants': [], 'scores': [], 'edits': []}
                else:
                    result = {'reactants': [], 'scores': [], 'edits': []}

                # 计算编辑距离
                if len(result['reactants']) > 0:
                    if i == 1:   # 第一步 此时产物就是终产物
                        mapped_targets = []
                        infers = []
                        edit_dists = []
                        for r_smi in copy.deepcopy(result['reactants']):
                            try:
                                mapped_r, mapped_p = map_step_rxn(r_smi, target_mol, mapper)
                                mapped_r = remap_smi_according_to_infer(mapped_p, mapped_p, mapped_r)
                                dist = calculate_edits_distance(mapped_p, mapped_r)
                                mapped_targets.append(mapped_p)
                                infers.append(mapped_r)
                                edit_dists.append(dist)
                            except Exception as e:
                                result['scores'].pop(result['reactants'].index(r_smi))
                                result['edits'].pop(result['reactants'].index(r_smi))
                                result['reactants'].remove(r_smi)

                    else: # 第二步往后
                        mapped_targets = []
                        infers = []
                        edit_dists = []
                        for r_smi in copy.deepcopy(result['reactants']):
                            try:
                                mapped_r, mapped_p = map_step_rxn(r_smi, m_next.mol, mapper)
                                mapped_r = remap_smi_according_to_infer(m_next.parent.infer, mapped_p, mapped_r)
                                dist = calculate_edits_distance(m_next.parent.mapped_target, mapped_r)
                                infers.append(mapped_r)
                                mapped_targets.append(m_next.parent.mapped_target)
                                edit_dists.append(dist)
                            except Exception as e:
                                result['scores'].pop(result['reactants'].index(r_smi))
                                result['edits'].pop(result['reactants'].index(r_smi))
                                result['reactants'].remove(r_smi)

                    reacts_lst = [react.split(".") for react in result['reactants']]
                    assert len(reacts_lst) == len(infers) == len(edit_dists) == len(mapped_targets)

                    # 归一化当前组别的编辑距离; 用于计算引导函数（反应成本函数）
                    costs = args.cost_weight * np.array(result['scores']) + (
                            1 - args.cost_weight) * normalize_edit_dist(edit_dists)
                    costs = 0.0 - np.log(np.clip(costs, 1e-3, 1.0))    # 负自然对数

                    succ, succ_num = mol_tree.expand(m_next, reacts_lst, costs, result['edits'],
                                                     infers, edit_dists, mapped_targets, result['scores'])

                else:
                    succ, succ_num = mol_tree.expand(
                        m_next, [], [], [], [], [], [], [])
                    logging.info('Expansion fails on %s!' % m_next.mol)

                if succ_num == 1:  # 统计首条路线搜索时间
                    time_cost_f = time.time() - start_time

                if succ and succ_num >= max_routes_num:
                    break

            time_cost = time.time() - start_time
            logging.info('Final routes num | time cost | iter: %s | %s | %d'
                         % (str(mol_tree.succ_num), str(time_cost), i))

            if mol_tree.succ:
                total_routes = mol_tree.get_routes_set()  # 所有路线集合
                score_best = mol_tree.get_score_best_route(total_routes)
                len_best = mol_tree.get_len_best_route(total_routes)
                if args.mode == "find_param":
                    multi_process_best_set = {}
                    filter_ratio = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
                    coef = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                    param_combin = product(filter_ratio, coef)
                    for (filter_ratio, coef) in param_combin:
                        multi_process_best_set[(filter_ratio, coef)] = mol_tree.get_multi_process_best_route(
                            total_routes, filter_ratio, coef
                        )
                else:
                    multi_process_best = mol_tree.get_multi_process_best_route(
                        total_routes, args.filter_ratio, args.coef
                        )
                    multi_process_best_set = {(args.filter_ratio, args.coef): multi_process_best}
            else:   ## 没找到路线
                total_routes = []
                score_best = None
                len_best = None
                multi_process_best_set = {}

        else:  ## 产物在化合物库中，不需要找路线
            time_cost = time.time() - start_time
            logging.info('Final routes num | time cost | iter: %s | %s | %d'
                         % (str(mol_tree.succ_num), str(time_cost), i))
            total_routes = []
            score_best = None
            len_best = None
            multi_process_best_set = {}

        if time_cost_f is None:
            time_cost_f = time_cost
        return mol_tree.succ_num, (time_cost, time_cost_f), i, total_routes, score_best, len_best, multi_process_best_set
    except:
        return 0, (0, 0), 0, [], None, None, {}
