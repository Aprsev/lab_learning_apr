#/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :CNMF.py
# @Time      :2022/1/11 19:43
# @Author    :PangXZ
import numpy as np


def cnmf(X, C, r, k, e):
    '''
    参数描述
    :param X: 原始矩阵，维度为d*n
    :param C: 有标签样本指示矩阵，维度为l*c (l：有标签的样本的数量，c：类别数量)
    :param r: 分解的两个非负矩阵的隐变量维度，要远小于原始矩阵的维度
    :param k: 迭代次数
    :param e: 理想误差
    :return: U:基矩阵 V：系数矩阵
    '''
    d, n = X.shape
    l, c = C.shape

    # 计算A矩阵
    I = np.mat(np.identity(n - l))
    A = np.zeros((n, n + c - l))

    for i in range(l):
        for j in range(c):
            A[i, j] = C[i, j]
    for q in range(n - l):
        A[l + q, c + q] = I[q, q]
    A = np.mat(A)
    U = np.mat(np.random.rand(d, r))
    Z = np.mat(np.random.rand(n + c - l, r))

    x = 1
    for x in range(k):
        print('-------------------------------------------------')
        print('开始第', x, '轮迭代')
        X_pre = U * (A * Z).T
        E = X - X_pre
        # print(E)
        err = 0.0
        for i in range(d):
            for j in range(n):
                err += E[i, j] * E[i, j]
        print('误差：', err)

        if err < e:
            break
        # update U
        a_u = U * Z.T * A.T * A * Z
        b_u = X * A * Z
        for i in range(d):
            for j in range(r):
                if a_u[i, j] != 0:
                    U[i, j] = U[i, j] * b_u[i, j] / a_u[i, j]
        # print(U)

        # update Z
        # print(Z.shape,n,r)
        a_z = A.T * A * Z * U.T * U
        b_z = A.T * X.T * U
        for i in range(n + c - l):
            for j in range(r):
                if a_z[i, j] != 0:
                    Z[i, j] = Z[i, j] * b_z[i, j] / a_z[i, j]
        # print(Z)
        print('第', x, '轮迭代结束')

    V = (A * Z).T
    return U, V


if __name__ == "__main__":
    X = [[5, 3, 2, 1, 2, 3],
         [4, 2, 2, 1, 1, 5],
         [1, 1, 2, 5, 2, 3],
         [1, 2, 2, 4, 3, 2],
         [2, 1, 5, 4, 1, 1],
         [1, 2, 2, 5, 3, 2],
         [2, 5, 3, 2, 2, 5],
         [2, 1, 2, 5, 1, 1], ]  # 8*6,6个样本
    X = np.mat(X)
    C = [[0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0], ]  # 4*3，假设有4个样本有标签，总共有三类标签
    C = np.mat(C)
    r=int(input("Please input your set r of cNMF:"))
    U, V = cnmf(X, C, r, 100, 0.01)
    print(U.shape, V.shape)
    print(U * V)
    input()