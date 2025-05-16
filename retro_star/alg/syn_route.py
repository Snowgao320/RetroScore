import numpy as np
from queue import Queue
from graphviz import Digraph

class SynRoute:
    def __init__(self, target_mol, succ_value):
        self.target_mol = target_mol
        self.mols = [target_mol]
        self.values = [None]  # succ_value
        self.Edits = [None]
        self.parents = [-1]
        self.children = [None]
        self.costs = {}
        self.diffs = {}
        self.total_diffs = {}
        self.end_total_diff = None

        self.succ_value = succ_value
        self.total_cost = 0
        self.sum_diff = 0
        self.total_score = 0
        self.length = 0

    def _add_mol(self, mol, parent_id):
        self.mols.append(mol)
        self.values.append(None)
        self.Edits.append(None)
        self.parents.append(parent_id)
        self.children.append(None)

        self.children[parent_id].append(len(self.mols)-1)

    def set_value(self, mol, value):
        assert mol in self.mols

        mol_id = self.mols.index(mol)
        self.values[mol_id] = value

    def add_reaction(self, mol, value, template, reactants, cost, diff, total_diff, score):
        assert mol in self.mols

        self.total_cost += cost
        self.sum_diff += diff
        self.length += 1
        self.total_score += score

        parent_id = self.mols.index(mol)
        self.values[parent_id] = value
        self.Edits[parent_id] = template
        self.children[parent_id] = []
        self.costs[parent_id] = cost
        self.diffs[parent_id] = diff
        self.total_diffs[parent_id] = total_diff

        for reactant in reactants:
            self._add_mol(reactant, parent_id)

    def viz_route(self, viz_file):
        G = Digraph('G', filename=viz_file)
        G.attr('node', shape='box')
        G.format = 'pdf'

        names = []
        for i in range(len(self.mols)):
            name = self.mols[i]
            # if self.templates[i] is not None:
            #     name += ' | %s' % self.templates[i]
            names.append(name)

        node_queue = Queue()
        node_queue.put((0,-1))   # target mol idx, and parent idx
        while not node_queue.empty():
            idx, parent_idx = node_queue.get()

            if parent_idx >= 0:
                G.edge(names[parent_idx], names[idx], label='cost')

            if self.children[idx] is not None:
                for c in self.children[idx]:
                    node_queue.put((c, idx))

        G.render()

    def serialize_reaction(self, idx):
        s = self.mols[idx]  # 终产物
        if self.children[idx] is None:
            return s  # root 没有子节点
        s += '>%.4f>' % np.exp(-self.costs[idx])
        s += self.mols[self.children[idx][0]]
        for i in range(1, len(self.children[idx])):
            s += '.'
            s += self.mols[self.children[idx][i]]

        return s

    def serialize(self, need_action=False):
        s = self.serialize_reaction(0)
        for i in range(1, len(self.mols)):
            if self.children[i] is not None:
                s += '|'
                s += self.serialize_reaction(i)

        if need_action:
            s += ';'
            s += str(self.Edits)
        return s
