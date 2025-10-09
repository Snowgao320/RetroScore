import math
import numpy as np
from queue import Queue
import logging
from retro_star.alg.mol_node import MolNode
from retro_star.alg.reaction_node import ReactionNode
from retro_star.alg.syn_route import SynRoute
import copy


class MolTree:
    def __init__(self, N, target_mol, known_mols, value_fn, zero_known_value=True):
        self.N = N
        self.target_mol = target_mol  # 目标产物smi
        self.known_mols = known_mols  # 已知化合物库
        self.value_fn = value_fn   # 分子合成成本估计网络
        self.zero_known_value = zero_known_value  # 是否指定在库分子init value == 0
        self.mol_nodes = []  # 所有分子节点
        self.reaction_nodes = []  # 所有反应节点

        self.root = self._add_mol_node(target_mol, None)
        self.succ = target_mol in known_mols  # 表明是否找到路线

        self.succ_num = 0  # 找到路线数量
        self.mol_in_target = False

        if self.succ:
            logging.info('Synthesis route found: target in starting molecules')
            self.succ_num = -1
            self.mol_in_target = True

    def check_route_succ(self, r_node):
        ancestors = r_node.get_ancestors_succ()
        return all(ancestors)

    def _add_mol_node(self, mol, parent):
        is_known = mol in self.known_mols
        init_value = self.value_fn.value_fn_run(mol)

        mol_node = MolNode(
            mol=mol,
            init_value=init_value,
            parent=parent,
            is_known=is_known,
            zero_known_value=self.zero_known_value
        )
        # 树添加分子节点
        self.mol_nodes.append(mol_node)
        mol_node.id = len(self.mol_nodes)
        return mol_node

    def _add_reaction_and_mol_nodes(self, cost, mols, parent, template, ancestors, infer, dist, target, score):
        # 添加单个反应节点和其子mol
        for mol in mols:  # 已经出现的分子不再添加
            if mol in ancestors:
                return None

        # 添加反应节点 parent, cost, template, infer, total_diff
        reaction_node = ReactionNode(parent, cost, template, infer, dist, target, score)
        # 添加该反应节点的子mol节点
        for mol in mols:
            self._add_mol_node(mol, reaction_node)

        reaction_node.init_values()
        self.reaction_nodes.append(reaction_node)
        reaction_node.id = len(self.reaction_nodes)
        return reaction_node

    def expand(self, mol_node, reacts, costs, templates, inference, dists, targets, scores):
        assert not mol_node.is_known and not mol_node.children

        if len(reacts) == 0: # 没有单步预测结果
            print('no one step model pred result!')
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
            return self.succ, self.succ_num

        assert mol_node.open
        ancestors = mol_node.get_ancestors()  # list including self and all parent_mol_node

        new_r_n = []
        for i in range(len(reacts)):
            r_n = self._add_reaction_and_mol_nodes(
                costs[i], reacts[i], mol_node, templates[i], ancestors,
                inference[i], dists[i], targets[i], scores[i]
            )
            if r_n is not None:
                new_r_n.append(r_n)

        if len(mol_node.children) == 0:      # No valid expansion results
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
            return self.succ, self.succ_num

        v_delta = mol_node.init_values()
        if mol_node.parent:
            mol_node.parent.backup(v_delta, from_mol=mol_node.mol)

        # 更新找到的路线的数量
        for n in new_r_n:
            if n.succ and self.check_route_succ(n):
                self.succ_num += 1

        if not self.succ and self.root.succ:
            logging.info('Synthesis route found!')
            self.succ = True

        return self.succ, self.succ_num

    def get_routes_set(self):
        # 输出找到的所有路线集合
        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value
        )

        total_routes = []
        for num in range(self.succ_num):
            _syn_route = copy.deepcopy(syn_route)
            mol_queue = Queue()
            mol_queue.put(self.root)
            queued_mol = []
            while not mol_queue.empty():
                mol = mol_queue.get()
                queued_mol.append(mol)
                if mol.is_known:
                    _syn_route.set_value(mol.mol, mol.succ_value)
                    continue

                current_reaction = None          # 选取一个反应节点
                for reaction in mol.children:
                    if reaction.succ and current_reaction is None:
                        current_reaction = reaction
                    elif reaction.succ and reaction.succ_value < current_reaction.succ_value:
                        current_reaction = reaction

                if current_reaction is None:   # 说明没有成功反应节点，该路线不通
                    break

                reactants = []
                for node in current_reaction.children:
                    mol_queue.put(node)
                    reactants.append(node.mol)

                _syn_route.add_reaction(
                    mol = mol.mol,
                    value = mol.succ_value,
                    template = current_reaction.template,
                    reactants = reactants,
                    cost = current_reaction.cost,
                    dist = current_reaction.edit_dist,
                    step_dist = current_reaction.step_dist,
                    score = current_reaction.confidence_score
                )

            end_reaction = queued_mol[-1].parent
            end_reaction.succ = False
            end_reaction.parent.search_backup()
            _syn_route.end_dist = end_reaction.edit_dist

            total_routes.append(_syn_route)

            if not self.root.succ:
                break

        return total_routes

    def get_multi_process_best_route(self, routes, filter_ratio, coef):
        ## 先根据average score取出 <= filter_ratio 条路线
        routes = sorted(routes, key=lambda x: x.sum_scores, reverse=True)
        if len(routes) == 1:
            return routes[0]

        threshold = round(len(routes) * filter_ratio)
        if threshold <= 1:
            return routes[0]

        routes = routes[:threshold]
        routes.sort(key=lambda x: x.end_dist, reverse=False)
        ## 根据极差系数进行规整，使得这里的end_dist相差不大
        if len(routes) < 2:
            stop = True
        else:
            stop = False

        while not stop:
            if len(routes) < 2:
                break
            max_min = routes[-1].end_dist - routes[0].end_dist
            aver = np.average(np.array([r.end_dist for r in routes]))
            if max_min/aver >= coef:
                routes.remove(routes[-1])
            else:
                stop = True

        # 根据step dist均值选出best
        routes.sort(key=lambda x: x.sum_step_dists/x.length, reverse=True)
        return routes[0]

    def get_score_best_route(self, routes: list):
        # 得到置信度分数最高的路线
        routes = sorted(routes, key=lambda x: x.sum_scores, reverse=True)
        return routes[0]

    def get_len_best_route(self, routes: list):
        # 得到置信度分数最高的路线
        routes = sorted(routes, key=lambda x: x.length, reverse=False)
        return routes[0]

