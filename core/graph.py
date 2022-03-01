# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:25:06 2019

@author: zhangjuefei
"""


class Graph:
    """
    计算图类
    """

    def __init__(self):
        self.nodes = []  # 计算图内的节点的列表
        self.name_scope = None

    def add_node(self, node):
        """
        添加节点
        """
        self.nodes.append(node)

    def clear_jacobi(self):
        """
        清除图中全部节点的雅可比矩阵
        """
        for node in self.nodes:
            node.clear_jacobi()

    def reset_value(self):
        """
        重置图中全部节点的值
        """
        for node in self.nodes:
            node.reset_value(False)  # 每个节点不递归清除自己的子节点的值

    def node_count(self):
        return len(self.nodes)


# 全局默认计算图
default_graph = Graph()
