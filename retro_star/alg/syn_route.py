import numpy as np
from queue import Queue

class SynRoute:
    def __init__(self, target_mol, succ_value):
        self.target_mol = target_mol   # 目标产物
        self.mols = [target_mol]       # 路线涉及所有分子
        self.values = [None]           # 每个mol node succ_value
        self.Edits = [None]            # 每步使用的模板（编辑序列）
        self.children = [None]         # parent id 位置为list（children id）
        self.costs = {}               # parent id: reaction cost
        self.dists = {}               # parent id: edit dist
        self.step_dists = {}          # parent id: step dist
        self.end_dist = None          # edit dist of end step

        self.succ_value = succ_value
        self.sum_scores = 0
        self.sum_step_dists = 0
        self.sum_costs = 0
        self.length = 0

    def _add_mol(self, mol, parent_id):
        self.mols.append(mol)
        self.values.append(None)
        self.Edits.append(None)
        self.children.append(None)
        self.children[parent_id].append(len(self.mols)-1)

    def set_value(self, mol, value):
        assert mol in self.mols
        mol_id = self.mols.index(mol)
        self.values[mol_id] = value

    def add_reaction(self, mol, value, template, reactants, cost, dist, step_dist, score):
        assert mol in self.mols
        self.sum_costs += cost
        self.sum_step_dists += step_dist
        self.length += 1
        self.sum_scores += score

        parent_id = self.mols.index(mol)
        self.values[parent_id] = value
        self.Edits[parent_id] = template
        self.children[parent_id] = []
        self.costs[parent_id] = cost
        self.dists[parent_id] = dist
        self.step_dists[parent_id] = step_dist

        for reactant in reactants:
            self._add_mol(reactant, parent_id)

    def serialize_reaction(self, idx):
        p = self.mols[idx]  # 产物
        if self.children[idx] is None:
            return p  # root 没有子节点

        p += ">>"
        p += self.mols[self.children[idx][0]]
        for i in range(1, len(self.children[idx])):
            p += '.'
            p += self.mols[self.children[idx][i]]
        return p

    def serialize(self, need_action=False):
        s = self.serialize_reaction(0)
        if need_action:
            s += ';'
            s += str(self.Edits[0])

        for i in range(1, len(self.mols)):
            if self.children[i] is not None:
                s += '|'
                s += self.serialize_reaction(i)

            if need_action:
                s += ';'
                s += str(self.Edits[i])

        return s
