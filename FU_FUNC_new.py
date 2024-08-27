import math
import numpy as np
import random

def fitness_fun(bij, Zj, A=200, R=0.8, t=10):
    nodes = 5  # 节点数量
    W = 20000  # 带宽20Mhz
    pij = 0.1  # 传输功率0.1w
    a1 = -10  # 噪声功率-100dB
    gij = 10  # 信道增益

    def Oij(pij, nodes, a1):
        gij = 10  # 信道增益
        Oij = math.log2(1 + ((pij * gij ** 2) / a1 ** 2))
        return Oij

    def cij(nij, W, Oij_value):
        epsilon = 1e-8  # 添加一个小常数以避免除以零
        return nij * W * Oij_value + epsilon

    # 定义相关性系数矩阵
    Qij = np.array([[0.6, 0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.3, 0.4, 0.5, 0.7],
                    [0.5, 0.1, 0.2, 0.3, 0.8],
                    [0.7, 0.3, 0.5, 0.4, 0.1],
                    [0.9, 0.5, 0.2, 0.1, 0.3]])

    # 设置相关性阈值
    T1 = 0.5
    # 生成nij
    nij = np.zeros_like(Qij)

    for i in range(5):
        sum_nij = 0
        for j in range(5):
            nij[i, j] = round(random.uniform(0, 1), 3)
            sum_nij += nij[i, j]
        if sum_nij > 1:
            nij[i] = nij[i] / sum_nij
        nij[i] = np.round(nij[i], 3)

    def Aj():
        Aj = 0
        for i in range(5):
            for j in range(5):
                if bij[i, j] > 0:
                    Aj += bij[i, j]
        return Aj

    def Tij(dj=10, t=10 , L=300):  # 时延
        aj = Zj / B
        Tj1 = t * dj * Aj() / aj
        Tj2 = dj * L / cij(nij, W, Oij_value)
        Tj2_max = np.max(Tj2, axis=1)
        Tij = Tj1 + Tj2_max
        Tij = np.sum(Tij, axis=0)
        return Tij

    def calculate_fi_for_all_nodes(nodes, t, s1=0.01, Zi=2, Zj=5, a2=0.0005, v=0.7):
        fi_values = {}
        for i in nodes:
            fi = 1 - np.exp(-s1 * Zi * (R * A * (t) ** a2) ** v)
            fj = 1 - np.exp(-s1 * Zj * (R * A * (t) ** a2) ** v)
            fi_values[(i, 'i')] = fi
            fi_values[(i, 'j')] = fj
        return fi_values

    e = 0.001  # 有效电容系数
    B = 0.15  # 处理数据的CPU周期数，单位Gcycles
    Zi = 2  # 请求节点平均计算能力
    nodes = 5  # 节点数量

    nodes = list(range(1, 6))
    fi_values = calculate_fi_for_all_nodes(nodes, t=t)
    Oij_values = [Oij(pij, nodes, a1) for i in range(5)]
    cij_values = [cij(nij[i, j], W, Oij_values[j]) for j in range(5) for i in range(5)]
    for i in range(5):
        for j in range(5):
            Oij_value = Oij_values[j]
            cij_value = cij_values[i * 5 + j]

    def Eij(Eth=50):
        Ey = round(random.uniform(15, 200), 1)
        e = 0.01
        dj = 10
        L = 300
        t = 10
        A = 100
        B = 0.15
        pij = 0.1  # 传输功率0.1w
        Ejt = e * t * B * A * Zj ** 2
        Tj2 = dj * L / cij(nij, W, Oij_value)
        Ejc = np.sum(pij * Tj2, axis=0)
        Ec = Ejc + Ejt
        Ej_value = Ec - max(Ey - Eth, 0)
        Eij = np.sum(Ej_value, axis=0)
        return Eij

    def Fj():
        Fj_values = np.sum(bij * Qij * fi_values[(i, 'j')]) / np.sum(bij * Qij)
        return Fj_values

    def ei(s1=10, Zi=2):
        return 1 - math.exp(-s1 * Zi * Fj())

    def X1(p1=20):
        Xi = 0
        for i in range(5):
            X1 = p1 * ei(s1=10, Zi=2)
            Xi += X1
        return Xi

    def FU():
        total_Uij = 0
        Zi = 2  # i的计算能力
        ei_value = ei(s1=10, Zi=2)
        p1 = 10  # 性能增益系数
        p2 = 0.5  # 能耗权重
        p3 = 0.4  # 时延权重
        Xi = X1(p1=20)  # 聚合模型收益
        L = 300
        Eij_value = Eij(Eth=50)
        Tij_value = Tij(dj=10, t=t, L=300)
        X2 = p2 * Eij_value
        X3 = p3 * Tij_value
        Ui = Xi - X2 - X3
        total_Uij += Ui
        return total_Uij

    total_Ui = FU()
    return total_Ui

# # FU_FUNC_new.py
# import math
# import numpy as np
# import random
#
# def fitness_fun(bij, Zj, A=200, R=0.8, t=10):
#     nodes = 5
#     W = 20000
#     pij = 0.1
#     a1 = -10
#     gij = 10
#     e = 0.001  # 有效电容系数
#     B = 0.15  # 处理数据的CPU周期数，单位Gcycles
#
#     def Oij(pij, nodes, a1):
#         gij = 10
#         return math.log2(1 + ((pij * gij ** 2) / a1 ** 2))
#
#     def cij(nij, W, Oij_value):
#         epsilon = 1e-8
#         return nij * W * Oij_value + epsilon
#
#     Qij = np.array([[0.6, 0.1, 0.2, 0.3, 0.4],
#                     [0.2, 0.3, 0.4, 0.5, 0.7],
#                     [0.5, 0.1, 0.2, 0.3, 0.8],
#                     [0.7, 0.3, 0.5, 0.4, 0.1],
#                     [0.9, 0.5, 0.2, 0.1, 0.3]])
#
#     T1 = 0.5
#     nij = np.zeros_like(Qij)
#
#     for i in range(5):
#         sum_nij = 0
#         for j in range(5):
#             nij[i, j] = round(random.uniform(0, 1), 3)
#             sum_nij += nij[i, j]
#         if sum_nij > 1:
#             nij[i] = nij[i] / sum_nij
#         nij[i] = np.round(nij[i], 3)
#
#     def Aj():
#         return np.sum(bij)
#
#     def Tij(dj=10, t=10, L=300):
#         aj = Zj / B
#         Tj1 = t * dj * Aj() / aj
#         Tj2 = dj * L / cij(nij, W, Oij(pij, nodes, a1))
#         Tj2_max = np.max(Tj2, axis=1)
#         return np.sum(Tj1 + Tj2_max)
#
#     def Eij(Eth=50):
#         Ey = round(random.uniform(15, 200), 1)
#         Ejt = e * t * B * A * Zj ** 2
#         Tj2 = 10 * 300 / cij(nij, W, Oij(pij, nodes, a1))
#         Ejc = np.sum(pij * Tj2, axis=0)
#         Ec = Ejc + Ejt
#         Ej_value = Ec - max(Ey - Eth, 0)
#         return np.sum(Ej_value, axis=0)
#
#     def FU():
#         total_Uij = 0
#         ei_value = 1 - math.exp(-10 * 2 * 1)
#         Xi = 20 * ei_value
#         X2 = 0.5 * Eij()
#         X3 = 0.4 * Tij()
#         Ui = Xi - X2 - X3
#         total_Uij += Ui
#         return total_Uij
#
#     return FU()
