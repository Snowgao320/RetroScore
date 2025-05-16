import numpy as np
import logging


class ReactionNode:
    def __init__(self, parent, cost, template, infer, total_diff, mapped_target, score):
        # 连接关系
        self.parent = parent
        parent.children.append(self)
        self.children = []

        # 在树中的位置（深度、树的反应节点list的索引）
        self.depth = self.parent.depth + 1
        self.id = -1  # 扩展后更新

        # 计算编辑距离
        self.mapped_target = mapped_target
        self.infer = infer  # 上一步反应步骤中所有中间体的mapped smi
        self.total_diff = total_diff  # 当前step对p_mol的总距离
        self.diff = self.compute_diff(total_diff) # 当前step和上一步的总距离之差

        # 单步模型输出结果
        self.cost = cost  # 单步置信度
        self.template = template  # 单步编辑详情
        self.score = score

        # Value
        self.value = None   # cost + 所有子mol的value和
        self.succ_value = np.inf    # sub solution value summation, only when succ
        self.target_value = None    # V_target(self | ROOT)

        # state
        self.succ = None    # successfully found a valid synthesis route
        self.open = True

    def get_ancestors_succ(self):
        if self.parent.parent is None:
            return [self.succ]

        ancestors = self.parent.parent.get_ancestors_succ()
        ancestors.append(self.succ)
        return ancestors

    def compute_diff(self, total_diff):
        if self.parent.parent is None:
            diff = total_diff
        else:
            diff = total_diff - self.parent.parent.total_diff
        return diff

    def v_self(self):
        """
        :return: V_self(self | subtree)
        """
        return self.value

    def v_target(self):
        """
        :return: V_target(self | whole tree)
        """
        return self.target_value

    def init_values(self):
        assert self.open
        self.value = self.cost
        self.succ = True
        for mol in self.children:
            self.value += mol.value
            self.succ &= mol.succ

        if self.succ:
            self.succ_value = self.cost
            for mol in self.children:
                self.succ_value += mol.succ_value

        self.target_value = self.parent.v_target() - self.parent.v_self() + self.value
        self.open = False

    def backup(self, v_delta, from_mol=None):
        self.value += v_delta
        self.target_value += v_delta

        self.succ = True
        for mol in self.children:
            self.succ &= mol.succ

        if self.succ:
            self.succ_value = self.cost
            for mol in self.children:
                self.succ_value += mol.succ_value

        if v_delta != 0:
            assert from_mol
            self.propagate(v_delta, exclude=from_mol)

        return self.parent.backup()

    def search_backup(self):
        assert self.succ == True

        new_succ = True
        for mol in self.children:
            new_succ &= mol.succ
        if self.succ != new_succ:
            self.succ = False
        return self.parent.search_backup()

    def propagate(self, v_delta, exclude=None):
        if exclude is None:
            self.target_value += v_delta

        for child in self.children:
            if exclude is None or child.mol != exclude:
                for grandchild in child.children:
                    grandchild.propagate(v_delta)

    def serialize(self):
        # return '%d' % (self.id)
        return '%d | value %.2f | target %.2f' % \
               (self.id, self.v_self(), self.v_target())
