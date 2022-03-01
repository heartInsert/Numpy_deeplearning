import numpy as np
from abc import abstractmethod
from .graph import default_graph


class Node(object):
    def __init__(self, *parents, **kargs):
        self.parents = list(parents)
        self.children = []
        self.jacobi = None
        self.value = None
        self.graph = kargs.get('graph', default_graph)
        for p in self.parents:
            p.add_child(self)

    def forward(self):
        for p in self.parents:
            if p.value is None:
                p.forward()
        self.compute()

    @abstractmethod
    def compute(self):
        '''前向传播'''
        pass

    # 静态图和动态图怎么区分？
    def backward(self, child_result):
        if self.jacobi is None:
            if self is child_result:
                '''如果自己是结果节点,导数为1,递归的终点'''
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                '''返回结果对孩子节点的导数*孩子节点对自己的导数'''
                self.jacobi = np.mat(np.zeros(self.dimension()))
                for c in self.children:
                    self.jacobi += c.backward(child_result) * c.get_jacobi(self)
        return self.jacobi

    @abstractmethod
    def get_jacobi(self, parent):
        '''求自己对父节点的导数，会在父节点中调用'''
        pass

    def get_children(self):
        return self.children

    def add_child(self, child):
        self.children.append(child)

    def get_parents(self):
        return self.parents

    def shape(self):
        """
        返回本节点的值作为矩阵的形状：（行数，列数）
        """
        return self.value.shape

    def dimension(self):
        return self.value.shape[0] * self.value.shape[1]


class Variable(Node):
    '''
    参数节点和标签节点
    '''

    def __init__(self, dim, init=False, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

        # 如果需要初始化，则以正态分布随机初始化变量的值
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))

        # 变量节点是否参与训练
        self.trainable = trainable

    def set_value(self, value):
        """为变量赋值    """

        # assert isinstance(value, np.matrix) and value.shape == self.dim

        # 本节点的值被改变，重置所有下游节点的值
        # self.reset_value()
        self.value = value
        pass
