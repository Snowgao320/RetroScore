import copy
import torch
import numpy as np
import logging
from retro_star.alg.mol_tree import MolTree
from utils.calculate_edit_distance import (map_step_rxn, calculate_edits_distance,
                                           remap_smi_according_to_infer, normalize_total_diff)
from rdkit import Chem


def canonicalize(mol):
    tmp = Chem.RemoveHs(mol)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return tmp


def canonicalize_prod(p):
    p = copy.deepcopy(p)
    p_mol = Chem.MolFromSmiles(p)
    if p_mol is None:
        return None
    p_mol = canonicalize(p_mol)
    for atom in p_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    p = Chem.MolToSmiles(p_mol)
    return p


def molstar(target_mol,starting_mols, expand_fn, value_fn, iterations, max_routes_num, args):

    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn
    )

    i = -1

    if not mol_tree.succ:
        for i in range(0, iterations):
            if i % 100 == 0:
                print(f'itered {i} times!')
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

            # mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open
            # single retro pred model forward
            can_mol = canonicalize_prod(m_next.mol)
            if can_mol is not None:
                # result = expand_fn(can_mol)
                result = expand_fn.beam_model_run(can_mol)
                # print(result)
            else:
                result = {'reactants': [], 'scores':[], 'edits': []}

            # 单步模型输出整理为字典形式；包含反应物、反应得分、编辑步骤；缺一就为None
            reactants = result['reactants']
            scores = result['scores']
            edits = result['edits']
            # delete None reactant
            for rid, r in enumerate(reactants):
                if r is None or r == "":
                    reactants.remove(r)
                    scores.remove(scores[rid])
                    edits.remove(edits[rid])
            assert len(reactants) == len(scores) == len(edits)

            if len(reactants) != 0:

                # costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                # costs = costs.tolist()

                # deepcopy
                reactants_copy = copy.deepcopy(reactants)
                scores_copy = copy.deepcopy(scores)
                edits_copy = copy.deepcopy(edits)

                if i ==0: # 第一步 此时产物就是终产物
                    mapped_target_list = []
                    inference_list = []
                    total_diff_list = []
                    for r_id, r_smi in enumerate(reactants_copy):
                        try:
                            ma_smi1, ma_target_smi = map_step_rxn(r_smi, target_mol)
                            total_diff1 = calculate_edits_distance(ma_target_smi, ma_smi1)
                            mapped_target_list.append(ma_target_smi)
                            inference_list.append(ma_smi1)
                            total_diff_list.append(total_diff1)
                        except Exception as e:
                            print(e)
                            # print('mapped fail')
                            reactants.remove(r_smi)
                            scores.remove(scores_copy[r_id])
                            edits.remove(edits_copy[r_id])
                else: # 第二步往后
                    mapped_target_list = []
                    inference_list = []
                    total_diff_list = []
                    for r_id, r_smi in enumerate(reactants_copy):
                        try:
                            ma_smi, ma_pro_smi = map_step_rxn(r_smi, m_next.mol)
                            rm_r_smi, rm_p_smi = remap_smi_according_to_infer(m_next.parent.infer, ma_pro_smi, ma_smi)
                            total_diff = calculate_edits_distance(m_next.parent.mapped_target, rm_r_smi)
                            inference_list.append(rm_r_smi)
                            mapped_target_list.append(m_next.parent.mapped_target)
                            total_diff_list.append(total_diff)
                        except:
                            reactants.remove(r_smi)
                            scores.remove(scores_copy[r_id])
                            edits.remove(edits_copy[r_id])

                reactant_lists = []
                for j in range(len(reactants)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert len(reactant_lists) == len(edits) == len(inference_list) == len(total_diff_list) == len(mapped_target_list)
                # 归一化当前组别的距离
                cost_weight = args.cost_weight
                costs = cost_weight*np.array(scores) + (1-cost_weight)*normalize_total_diff(total_diff_list)
                costs = 0.0 - np.log(np.clip(np.array(costs), 1e-3, 1.0))

                succ, succ_num = mol_tree.expand(m_next, reactant_lists, costs, edits,
                                                 inference_list, total_diff_list, mapped_target_list, scores)

            else:
                succ, succ_num = mol_tree.expand(m_next, [], [], [], [], [], [], [])
                logging.info('Expansion fails on %s!' % m_next.mol)

            if succ and succ_num >= max_routes_num:
                break

        logging.info('Final search num | success value | iter: %s | %s | %d'
                     % (str(mol_tree.succ_num), str(mol_tree.root.succ_value), i+1))


    topk_routes = None
    best_cost_route = None
    best_diff_route = None
    best_len_route = None
    # best_diff_route_05 = None
    # best_diff_route_03 = None
    # best_diff_route_01 = None
    if mol_tree.succ and not mol_tree.mol_in_target:
        topk_routes = mol_tree.get_topk_routes()
        assert len(topk_routes) != 0
        best_diff_route = mol_tree.get_best_route(topk_routes, args.filter_radio, args.filter_coef)
        best_cost_route = mol_tree.get_cost_best_route(topk_routes)
        best_len_route = mol_tree.get_len_best_route(topk_routes)
        # best_diff_route_05 = mol_tree.get_best_route_5(topk_routes)
        # best_diff_route_03 = mol_tree.get_best_route_3(topk_routes)
        # best_diff_route_01 = mol_tree.get_best_route_1(topk_routes)


    return mol_tree.succ_num, (best_diff_route, i+1), best_cost_route, best_len_route, topk_routes
