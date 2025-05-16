import numpy as np
from queue import Queue
import logging
from retro_star.alg.mol_node import MolNode
from retro_star.alg.reaction_node import ReactionNode
from retro_star.alg.syn_route import SynRoute
import copy


class MolTree:
    def __init__(self, target_mol, known_mols, value_fn, zero_known_value=True):
        self.target_mol = target_mol  # 目标产物smi
        self.known_mols = known_mols  # 已知化合物库
        self.value_fn = value_fn   # 分子成本估计网络
        self.zero_known_value = zero_known_value  # 是否指定在库分子init value == 0
        self.mol_nodes = []  # 所有分子节点
        self.reaction_nodes = []  # 所有反应节点

        self.root = self._add_mol_node(target_mol, None)
        self.succ = target_mol in known_mols  # 表明是否找到路线
        # self.search_status = 0
        self.succ_num = 0  # 找到路线数量
        self.mol_in_target = False

        if self.succ:
            logging.info('Synthesis route found: target in starting molecules')
            self.succ_num = -1
            self.mol_in_target = True


    def check_all_parents(self, r_n):
        assert r_n.succ == True
        ancestors = r_n.get_ancestors_succ()
        if all(ancestors) == True:
            return True
        else:
            return False

    def _add_mol_node(self, mol, parent):
        is_known = mol in self.known_mols
        init_value = self.value_fn.value_fn_run(mol)
        # print('init_value:', init_value)

        # mol, init_value, parent = None, is_known = False, zero_known_value = True
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

    def _add_reaction_and_mol_nodes(self, cost, mols, parent, template, ancestors, infer, total_diff, mapped_target, score):
        # 添加单个反应节点和其子mol
        # print(cost)
        assert cost >= 0

        for mol in mols:  # 已经出现的分子不再添加
            if mol in ancestors:
                return None

        # 添加反应节点 parent, cost, template, infer, total_diff
        reaction_node = ReactionNode(parent, cost, template, infer, total_diff, mapped_target, score)
        # 添加该反应节点的子mol节点
        for mol in mols:
            self._add_mol_node(mol, reaction_node)

        reaction_node.init_values()
        self.reaction_nodes.append(reaction_node)
        reaction_node.id = len(self.reaction_nodes)

        return reaction_node

    def expand(self, mol_node, reactant_lists, costs, templates, inference, total_diffs, map_targets, scores):
        assert not mol_node.is_known and not mol_node.children

        if len(reactant_lists) == 0: # 没有单步预测结果
            print('no one step model pred result!')
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
            return self.succ, self.succ_num

        assert mol_node.open
        ancestors = mol_node.get_ancestors()  # dict including self and all self.parent_mol_node mols

        new_r_n = []
        for i in range(len(costs)):
            r_n = self._add_reaction_and_mol_nodes(costs[i], reactant_lists[i],
                                                    mol_node, templates[i], ancestors,
                                                    inference[i], total_diffs[i], map_targets[i], scores[i])
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
        for rn in new_r_n:
            if not rn.succ:
                continue
            if self.check_all_parents(rn):
                self.succ_num += 1
        # print('succ num')
        # print(self.succ_num)

        if not self.succ and self.root.succ:
            logging.info('Synthesis route found!')
            self.succ = True

        return self.succ, self.succ_num

    def get_topk_routes(self):
        # 输出topk路线和total cost最小的路线
        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value
        )

        topk_routes = []
        for i in range(0, self.succ_num):
            syn_route1 = copy.deepcopy(syn_route)
            mol_queue = Queue()
            mol_queue.put(self.root)
            queued_mol = []
            while not mol_queue.empty():
                mol = mol_queue.get()
                queued_mol.append(mol)
                if mol.is_known:
                    syn_route1.set_value(mol.mol, mol.succ_value)
                    continue

                # 选取一个反应节点
                current_reaction = None
                for reaction in mol.children:
                    if reaction.succ and current_reaction is None:
                        current_reaction = reaction

                    elif reaction.succ and reaction.succ_value < current_reaction.succ_value:
                        current_reaction = reaction

                reactants = []
                for reactant in current_reaction.children:
                    mol_queue.put(reactant)
                    reactants.append(reactant.mol)

                syn_route1.add_reaction(
                    mol=mol.mol,
                    value=mol.succ_value,
                    template=current_reaction.template,
                    reactants=reactants,
                    cost=current_reaction.cost,
                    diff=current_reaction.diff,
                    total_diff= current_reaction.total_diff,
                    score = current_reaction.score
                )

            end_reaction = queued_mol[-1].parent
            end_reaction.succ = False
            end_reaction.parent.search_backup()
            syn_route1.end_total_diff = end_reaction.total_diff

            topk_routes.append(syn_route1)
            if self.root.succ == False:
                break

        topk_routes.sort(key=lambda x: x.total_cost)
        return topk_routes

    def get_best_route(self, topk_routes, filter_radio, filter_coef):
        # 得到距离上的最佳路线
        ## 先根据cost取出<=50%条路线
        topk_routes_copy = copy.deepcopy(topk_routes)
        topk_routes_copy.sort(key=lambda x: x.total_score, reverse=True)
        total_diff_routes = []
        for i in range(len(topk_routes_copy)):
            total_diff_routes.append(topk_routes_copy[i])
            if len(total_diff_routes) >= round(filter_radio*len(topk_routes_copy)):
                break

        total_diff_routes.sort(key=lambda x: x.end_total_diff)
        ## 根据极差系数进行规整，使得这里的end total diff相差不大
        if len(total_diff_routes) < 2:
            stop = False
        else:
            stop = True
        while stop:
            if len(total_diff_routes) < 2:
                break
            max_min = total_diff_routes[-1].end_total_diff - total_diff_routes[0].end_total_diff
            aver = np.average(np.array([r.end_total_diff for r in total_diff_routes]))
            coef = max_min / aver
            if coef >= filter_coef:
                total_diff_routes.remove(total_diff_routes[-1])
            else:
                stop = False

        # 根据diff均值选出best
        sorted_routes = sorted(total_diff_routes, key=lambda x: x.sum_diff/x.length, reverse=True)
        return sorted_routes[0]

    def get_cost_best_route(self, topk_routes):
        # 得到成本最小的路线
        topk_routes_copy_1 = copy.deepcopy(topk_routes)
        topk_routes_copy_1.sort(key=lambda x: x.total_score, reverse=True)
        return topk_routes_copy_1[0]

    def get_len_best_route(self, topk_routes):
        # 得到长度最短的路线
        topk_routes_copy_2 = copy.deepcopy(topk_routes)
        topk_routes_copy_2.sort(key=lambda x: x.length)
        return topk_routes_copy_2[0]

    def viz_search_tree(self, viz_file): #TODO: 下载可视化包，测试函数
        G = Digraph('G', filename=viz_file)
        G.attr(rankdir='LR')
        G.attr('node', shape='box')
        G.format = 'pdf'

        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            node, parent = node_queue.get()

            if node.open:
                color = 'lightgrey'
            else:
                color = 'aquamarine'

            if hasattr(node, 'mol'):
                shape = 'box'
            else:
                shape = 'rarrow'

            if node.succ:
                color = 'lightblue'
                if hasattr(node, 'mol') and node.is_known:
                    color = 'lightyellow'

            G.node(node.serialize(), shape=shape, color=color, style='filled')

            label = ''
            if hasattr(parent, 'mol'):
                label = f'{node.cost};{node.total_diff};{node.succ_value}'

            if parent is not None:
                G.edge(parent.serialize(), node.serialize(), label=label)

            if node.children is not None:
                for c in node.children:
                    node_queue.put((c, node))

        G.render()


    def get_best_route_5(self, topk_routes):
        # 得到距离上的最佳路线
        ## 先根据cost取出<=50%条路线
        topk_routes_copy_5 = copy.deepcopy(topk_routes)
        topk_routes_copy_5.sort(key=lambda x: x.total_cost)
        total_diff_routes = []
        for i in range(len(topk_routes_copy_5)):
            total_diff_routes.append(topk_routes_copy_5[i])
            if len(total_diff_routes) >= round(0.3*len(topk_routes_copy_5)):
                break
        total_diff_routes.sort(key=lambda x: x.end_total_diff)
        ## 根据极差系数进行规整，使得这里的end total diff相差不大
        if len(total_diff_routes) < 2:
            stop = False
        else:
            stop = True
        while stop:
            if len(total_diff_routes) < 2:
                break
            max_min = total_diff_routes[-1].end_total_diff - total_diff_routes[0].end_total_diff
            aver = np.average(np.array([r.end_total_diff for r in total_diff_routes]))
            coef = max_min / aver
            if coef >= 1:
                total_diff_routes.remove(total_diff_routes[-1])
            else:
                stop = False

        # 根据diff均值选出best
        sorted_routes = sorted(total_diff_routes, key=lambda x: x.sum_diff/x.length, reverse=True)
        return sorted_routes[0]


    def get_best_route_3(self, topk_routes):
        # 得到距离上的最佳路线
        # 先根据cost取出<=50%条路线
        topk_routes_copy3 = copy.deepcopy(topk_routes)
        topk_routes_copy3.sort(key=lambda x: x.total_cost)
        total_diff_routes = []
        for i in range(len(topk_routes_copy3)):
            total_diff_routes.append(topk_routes_copy3[i])
            if len(total_diff_routes) >= round(0.3*len(topk_routes_copy3)):
                break
        total_diff_routes.sort(key=lambda x: x.end_total_diff)
        ## 根据极差系数进行规整，使得这里的end total diff相差不大
        if len(total_diff_routes) < 2:
            stop = False
        else:
            stop = True
        while stop:
            if len(total_diff_routes) < 2:
                break
            max_min = total_diff_routes[-1].end_total_diff - total_diff_routes[0].end_total_diff
            aver = np.average(np.array([r.end_total_diff for r in total_diff_routes]))
            coef = max_min / aver
            if coef >= 0.7:
                total_diff_routes.remove(total_diff_routes[-1])
            else:
                stop = False

        # 根据diff均值选出best
        sorted_routes = sorted(total_diff_routes, key=lambda x: x.sum_diff/x.length, reverse=True)
        return sorted_routes[0]

    def get_best_route_1(self, topk_routes):
        # 得到距离上的最佳路线
        ## 先根据cost取出<=50%条路线
        topk_routes_copy1 = copy.deepcopy(topk_routes)
        topk_routes_copy1.sort(key=lambda x: x.total_cost)
        total_diff_routes = []
        for i in range(len(topk_routes_copy1)):
            total_diff_routes.append(topk_routes_copy1[i])
            if len(total_diff_routes) >= round(0.3*len(topk_routes_copy1)):
                break
        total_diff_routes.sort(key=lambda x: x.end_total_diff)
        ## 根据极差系数进行规整，使得这里的end total diff相差不大
        if len(total_diff_routes) < 2:
            stop = False
        else:
            stop = True
        while stop:
            if len(total_diff_routes) < 2:
                break
            max_min = total_diff_routes[-1].end_total_diff - total_diff_routes[0].end_total_diff
            aver = np.average(np.array([r.end_total_diff for r in total_diff_routes]))
            coef = max_min / aver
            if coef >= 0.5:
                total_diff_routes.remove(total_diff_routes[-1])
            else:
                stop = False

        # 根据diff均值选出best
        sorted_routes = sorted(total_diff_routes, key=lambda x: x.sum_diff/x.length, reverse=True)
        return sorted_routes[0]
