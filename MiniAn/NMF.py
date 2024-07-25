#/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :NMF_demo3.py
# @Time      :2022/1/11 16:28
# @Author    :PangXZ
import numpy as np


def nmf(X, r, k, e):
    '''
    参数说明
    :param X: 原始矩阵
    :param r: 分解的两个非负矩阵的隐变量维度，要远小于原始矩阵的维度
    :param k: 迭代次数
    :param e: 理想误差
    :return: W：基矩阵， H: 系数矩阵
    '''
    d, n = X.shape
    W = np.mat(np.random.rand(d, r))
    H = np.mat(np.random.rand(n, r))

    x = 1
    for x in range(k):
        print('---------------------------------------------------')
        print('开始第', x, '轮迭代')
        X_pre = W * H.T
        E = X - X_pre
        err = 0.0
        for i in range(d):
            for j in range(n):
                err += E[i, j] * E[i, j]
        print('误差：', err)

        if err < e:
            break
        a_w = W * (H.T) * H
        b_w = X * H
        for p in range(d):
            for q in range(r):
                if a_w[p, q] != 0:
                    W[p, q] = W[p, q] * b_w[p, q] / a_w[p, q]
        a_h = H * (W.T) * W
        b_h = X.T * W
        print(r, n)
        for c in range(n):
            for g in range(r):
                if a_h[c, g] != 0:
                    H[c, g] = H[c, g] * b_h[c, g] / a_h[c, g]
        print('第', x, '轮迭代结束')
    return W, H


if __name__ == "__main__":
    X = [[5, 3, 2, 1, 2, 3],
         [4, 2, 2, 1, 1, 5],
         [1, 1, 2, 5, 2, 3],
         [1, 2, 2, 4, 3, 2],
         [2, 1, 5, 4, 1, 1],
         [1, 2, 2, 5, 3, 2],
         [2, 5, 3, 2, 2, 5],
         [2, 1, 2, 5, 1, 1], ]
    print(X)
    X = np.mat(X)
    r=int(input("Please input your set r of NMF:"))
    W, H = nmf(X, r, 100, 0.001)
    print(W)
    print(H)
    print(W * H.T)
    input()
