import numpy as np
import logging


class MolNode:
    def __init__(self, mol, init_value, parent=None, is_known=False, zero_known_value=True):
        # 连接关系
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
        self.children = []

        # 在树中的位置（深度、树的反应节点list的索引）
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth
        self.id = -1

        # 分子信息
        self.mol = mol  # smi

        # state
        self.is_known = is_known  # 表明分子是否在库
        self.succ = is_known  #表明该分子是否可得
        self.open = True  # before expansion: True, after expansion: False

        # value
        self.value = init_value  # value_fn(mol) 由模型给出
        self.succ_value = np.inf    # self.value , only when succ

        # 如果分子初始可得，重置init性质
        if is_known:
            self.open = False
            if zero_known_value:
                self.value = 0
            self.succ_value = self.value

    def v_self(self):
        """
        :return: V_self(self | subtree)
        """
        return self.value

    def v_target(self):
        """
        :return: V_target(self | root)
        """
        if self.parent is None:
            return self.value
        else:
            return self.parent.v_target()

    def init_values(self, no_child=False):
        # after expand, update mol node state,and return v_delta
        assert self.open and (no_child or self.children)

        new_value = np.inf
        new_succ = False
        for reaction in self.children:
            new_value = np.min((new_value, reaction.v_self()))
            new_succ |= reaction.succ

        if self.succ != new_succ:
            self.succ = new_succ

        v_delta = new_value - self.value
        self.value = new_value

        if self.succ:
            for reaction in self.children:
                self.succ_value = np.min((self.succ_value, reaction.succ_value))

        self.open = False

        return v_delta

    def backup(self):
        assert not self.is_known

        new_value = np.inf
        new_succ = False
        for reaction in self.children:
            new_value = np.min((new_value, reaction.v_self()))
            new_succ |= reaction.succ

        updated = (self.value != new_value) or (self.succ != new_succ)

        if updated:
            v_delta = new_value - self.value
            self.value = new_value
            self.succ = new_succ

            if self.succ:
                new_succ_value = np.inf
                for reaction in self.children:
                    new_succ_value = np.min((new_succ_value, reaction.succ_value))
                self.succ_value = new_succ_value

            if self.parent:
                return self.parent.backup(v_delta, from_mol=self.mol)

    def search_backup(self):
        succ = False
        for reacton in self.children:
            succ |= reacton.succ

        self.succ = succ

        if self.parent is not None:
            return self.parent.search_backup()

        # else:
        #     raise ValueError('search backup failed!')


    def serialize(self):
        text = '%d | %s' % (self.id, self.mol)
        # text = '%d | %s | pred %.2f | value %.2f | target %.2f' % \
        #        (self.id, self.mol, self.pred_value, self.v_self(),
        #         self.v_target())
        return text

    def get_ancestors(self):
        if self.parent is None:
            return [self.mol]

        ancestors = self.parent.parent.get_ancestors()
        ancestors.append(self.mol)
        return ancestors