from ..core import Node
import numpy as np


def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / \
           filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


class Operator(Node):
    '''
    第三种Node节点
    '''
    pass


class Add(Operator):
    def __init__(self, *parents, **kargs):
        super().__init__(*parents, **kargs)

    def compute(self):
        self.value = np.mat(np.zeros(self.parents[0].shape()))
        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):
        return np.mat(np.eye(self.dimension()))  # 矩阵之和对其中任一个矩阵的雅可比矩阵是单位矩阵


class MatMul(Operator):

    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        # 矩阵乘法
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent):
        """
        将矩阵乘法视作映射，求映射对参与计算的矩阵的雅克比矩阵。
        """
        # 很神秘，靠注释说不明白了
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(
                self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(
                parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class SoftMax(Operator):
    """
    SoftMax函数
    """

    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2  # 防止指数过大
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        我们不实现SoftMax节点的get_jacobi函数，
        训练时使用CrossEntropyWithSoftMax节点
        """
        raise NotImplementedError("Don't use SoftMax's get_jacobi")


class Step(Operator):

    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 1.0, 0.0))

    def get_jacobi(self, parent):
        return np.zeros(np.where(self.parents[0].value.A1 >= 0.0, 0.0, -1.0))
