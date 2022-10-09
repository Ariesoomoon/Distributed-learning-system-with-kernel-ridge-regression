import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold
import time
import random
import math
from random import shuffle


'''
1. Create data, X is uniform distribution on the (hyper-) cube [0,1]
'''
# generate y from g1, dimension = 1
def create_y_g1(data):
    y = np.empty(shape=(len(data)))
    for i in range(len(data)):
        if (data[i] >= 0) and (data[i] <= 0.5):
            y[i] = max(((1.0 - 2 * data[i]) ** 3), 0) * (1.0 + 6 * data[i])
        else:
            y[i] = 0
    return y


# generate y from g2, dimension = 5
def create_y_g2(data):
    y = np.empty(shape=(len(data)))
    for i in range(len(data)):
        norm_row = np.linalg.norm(data[i])
        if (norm_row >= 0) and (norm_row <= 1):
            y[i] = (max((1 - norm_row), 0) ** 5) * (1 + 5 * norm_row) + 0.2 * norm_row ** 2
        else:
            y[i] = 0.2 * norm_row ** 2
    return y


# sampling y, d = 1, 5
def sample(train_size, test_size, d):
    X_train = np.random.uniform(0.0, 1.0, train_size)
    X_test = np.random.uniform(0.0, 1.0, test_size)

    if d == 1:
        noise = np.random.normal(0, 0.1, train_size)
        y_train = create_y_g1(X_train) + noise
        y_test = create_y_g1(X_test)

    else:
        noise = np.random.normal(0, 0.1, train_size[0])
        y_train = create_y_g2(X_train) + noise
        y_test = create_y_g2(X_test)
    return X_train.shape, y_train.shape, X_train, y_train, X_test.shape, y_test.shape, X_test, y_test


# generate data for simulation
def generate_data(train, test, d):
    g1sample = sample(train, test, d)
    X_train, y_train, X_test, y_test = g1sample[2], g1sample[3], g1sample[6], g1sample[7]
    print('------------------------------From d = %s ------------------------------------------' % d)
    print('Train set X:%s, y:%s  |  Test set X:%s, y:%s' % (g1sample[0], g1sample[1], g1sample[4], g1sample[5]))
    return X_train, y_train, X_test, y_test


'''
2. divide data, training parameters both for global and local
'''
# divide data for same size
def mblocks(x, y, m):
    blocks = DataFrame(columns=('block_x', 'block_y'))
    for i in range(m):
        a = len(x) / m
        begin, end = int(a * i), int(a * (i + 1))
        blocks.loc[i] = [x[begin: end], y[begin: end]]
    return blocks


# divide data for different size
def mblocks_diff(x_tra, m):
    np.random.seed(0)
    blocks = {}
    start, endlast = 0, 0
    for i in range(m-1):
        b_size = int(len(x_tra)/m)
        end = np.random.randint(int(b_size*0.8), b_size)    # e.g.: b_size = 100, the choose randomly from [80, 100]
        end = end + endlast
        blocks[i] = [j for j in range(start, end, 1)]
        endlast, start = end, end

    blocks[i+1] = [j for j in range(end, len(x_tra), 1)]
    return blocks



'''
3. training parameter h --------------------------------------------------
'''
'''
3.1 training parameter h for NWK(gaussian),NWK(naive), NWK(Epanechnikov) by cv-5; Pay attention to replace the function here
'''
def cv_global_h(x, y, f, h, d):
    s = 2
    kf = KFold(n_splits=f)
    error = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        error_validation = GE_gaussiank(x_train, y_train, x_test, y_test, h, d, s)
        error.append(error_validation)
    return sum(error)/f


def global_parameter_h(x_tra, y_tra, f, d):
    cv_compares, h_opt1s = [], []
    p1, p2 = 0.25, 0.75
    #print('initial p1,p2:', p1, p2)
    for i in range(3, 13):
        cv_p1 = cv_global_h(x_tra, y_tra, f, p1, d)
        cv_p2 = cv_global_h(x_tra, y_tra, f, p2, d)

        if cv_p1 <= cv_p2:
            #print('optimal h: %s, tuning at: %s' % (p1, p2 - p1))
            cv_compare, h_opt1 = cv_p1, p1
            p1, p2 = p1-2**(-i), p1+2**(-i)

        else:
            cv_compare, h_opt1 = cv_p2, p2
            #print('optimal h: %s, tuning at: %s' % (p2, p2 - p1))
            p1, p2 = p2-2**(-i), p2+2**(-i)

        #print('p1,p2:', p1, p2)
        cv_compares.append(cv_compare)
        h_opt1s.append(h_opt1)

    index = cv_compares.index(min(cv_compares))
    h_opt = h_opt1s[index]
    return h_opt


# training parameter h for NWK(gaussian),NWK(naive), NWK(Epanechnikov) on local agents(or local blocks)
def opt_local_h(x_tra, y_tra, m, f, d, j):
    cv_compares, h_opt1s = [], []
    b = mblocks(x_tra, y_tra, m)
    b_x = np.array(b.loc[j]['block_x'])
    b_y = np.array(b.loc[j]['block_y'])
    p1, p2 = 0.25, 0.75

    for i in range(3, 13):
        cv_p1 = cv_global_h(b_x, b_y, f, p1, d)
        cv_p2 = cv_global_h(b_x, b_y, f, p2, d)
        if cv_p1 <= cv_p2:
            # print('optimal k: %s, tuning at: %s'% (p1, p2-p1))
            cv_compare, h_opt1 = cv_p1, p1
            p1, p2 = p1 - 2 ** (-i), p1 + 2 ** (-i)
        else:
            cv_compare, h_opt1 = cv_p2, p2
            # print('optimal k: %s, tuning at: %s'% (p2, p2-p1))
            p1, p2 = p2 - 2 ** (-i), p2 + 2 ** (-i)

        # print('p1,p2:', p1, p2)
        cv_compares.append(cv_compare)
        h_opt1s.append(h_opt1)
        # print('cv_compares', cv_compares)
        # print('h_opt1s', h_opt1s)

    index = cv_compares.index(min(cv_compares))
    h_opt = h_opt1s[index]
    return h_opt


def local_parameter_h(x_tra, y_tra, m_nwk, f, d):
    h_adapt_blocks = []  # Store the opt k
    for j in range(m_nwk):
        h0 = opt_local_h(x_tra, y_tra, m_nwk, f, d, j)
        print('when m = %s, block = %s, h_adapt = %s' % (m_nwk, j, h0))
        h_adapt_blocks.append(h0)
    # print('h_adapt of the last block = %s when m= %s' % (a, m_nwk))
    return h_adapt_blocks


# training parameter h for each blocks
def local_parameter_h_test(x_tra, y_tra, f, d, fen_num, gap):
    local_hs = {}                        # Store the opt h of each block for all fen_num
    for i in range(fen_num):
        m_nwk = (i + 1) * gap
        print('m_nwk:', m_nwk)
        local_h = local_parameter_h(x_tra, y_tra, m_nwk, f, d)
        local_h = np.array(local_h)
        local_h = np.squeeze(local_h)    # list object is not callable
        local_hs[i] = local_h            # Store the opt h of each block for fen_num = i
    return local_hs



'''
3.2 (1) training parameter k by cv-5 of KNN, d=1
'''
def cv_global_k(x, y, k, d):
    kf = KFold(n_splits=5)
    error = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        error_validation = GE_Knn(x_train, y_train, x_test, y_test, k, d)
        error.append(error_validation)
    return sum(error)/5


def global_parameter_k_d1(x_tra, y_tra, d):
    cv_compares, k_opt1s = [], []
    p1, p2 = 105, 315
    print('initial p1,p2:', p1, p2)
    for i in range(1,8):
        cv_p1 = cv_global_k(x_tra, y_tra, p1, d)
        cv_p2 = cv_global_k(x_tra, y_tra, p2, d)
        if cv_p1 <= cv_p2:
            print('optimal k: %s, tuning at: %s' % (p1, p2 - p1))
            cv_compare, k_opt1 = cv_p1, p1
            p1, p2 = p1-int(105/(2**i)), p1+int(105/(2**i))
        else:
            cv_compare, k_opt1 = cv_p2, p2
            print('optimal k: %s, tuning at: %s' % (p2, p2 - p1))
            p1, p2 = p2-int(105/(2**i)), p2+int(105/(2**i))

        print('p1,p2:', p1, p2)
        cv_compares.append(cv_compare)
        k_opt1s.append(k_opt1)

    index = cv_compares.index(min(cv_compares))
    k_opt = k_opt1s[index]
    return k_opt


def opt_local_k_d1(x_tra, y_tra, m, p1, p2, d, j):
    cv_compares, k_opt1s = [], []
    b = mblocks(x_tra, y_tra, m)
    b_x = np.array(b.loc[j]['block_x'])
    b_y = np.array(b.loc[j]['block_y'])

    if p2 <= len(b_y):
        # print('initial p1,p2, according to design:', p1, p2)
        for i in range(1, 6):
            cv_p1 = cv_global_k(b_x, b_y, p1, d)
            cv_p2 = cv_global_k(b_x, b_y, p2, d)
            if cv_p1 <= cv_p2:
                print('optimal k: %s, tuning at: %s'% (p1, p2-p1))
                cv_compare, k_opt1 = cv_p1, p1
                p1, p2 = p1 - math.ceil(30 / (2 ** i)), p1 + math.ceil(30 / (2 ** i))
            else:
                cv_compare, k_opt1 = cv_p2, p2
                print('optimal k: %s, tuning at: %s'% (p2, p2-p1))
                p1, p2 = p2 - math.ceil(30 / (2 ** i)), p2 + math.ceil(30 / (2 ** i))
            # print('p1,p2:', p1, p2)
            cv_compares.append(cv_compare)
            k_opt1s.append(k_opt1)

    else:
        p1 = int(0.25 * len(b_y))
        p2 = int(0.75 * len(b_y))
        # print('initial p1,p2, according to block size:', p1, p2)
        for i in range(1, 6):
            cv_p1 = cv_global_k(b_x, b_y, p1, d)
            cv_p2 = cv_global_k(b_x, b_y, p2, d)
            if cv_p1 <= cv_p2:
                # print('optimal k: %s, tuning at: %s'% (p1, p2-p1))
                cv_compare = cv_p1
                k_opt1 = p1
                p1, p2 = math.ceil(0.5 * p1), math.ceil(1.5 * p1)
            else:
                cv_compare = cv_p2
                k_opt1 = p2
                # print('optimal k: %s, tuning at: %s'% (p2, p2-p1))
                p1, p2 = math.ceil(p2 - 0.5 * p1), math.ceil(p2 + 0.5 * p1)

            # print('p1,p2:', p1, p2)
            cv_compares.append(cv_compare)
            k_opt1s.append(k_opt1)
            # print('cv_compares', cv_compares)
            # print('k_opt1s', k_opt1s)

    index = cv_compares.index(min(cv_compares))
    k_opt = k_opt1s[index]
    return k_opt


'''3.2 (2) training parameter k by cv-5 of KNN, d=5'''
def global_parameter_k_d5(x_tra, y_tra, d):
    cv_compares, k_opt1s = [], []
    p1, p2 = 20, 60
    print('initial p1,p2:', p1, p2)
    for i in range(5):
        cv_p1 = cv_global_k(x_tra, y_tra, p1, d)
        cv_p2 = cv_global_k(x_tra, y_tra, p2, d)
        if cv_p1 <= cv_p2:
            print('optimal k: %s, tuning at: %s' % (p1, p2 - p1))
            cv_compare, k_opt1 = cv_p1, p1
            p1, p2 = p1 - int(10 / (2 ** i)), p1 + int(10 / (2 ** i))
        else:
            cv_compare, k_opt1 = cv_p2, p2
            print('optimal k: %s, tuning at: %s' % (p2, p2 - p1))
            p1, p2 = p2 - int(10 / (2 ** i)), p2 + int(10 / (2 ** i))

        print('p1,p2:', p1, p2)
        cv_compares.append(cv_compare)
        k_opt1s.append(k_opt1)

    index = cv_compares.index(min(cv_compares))
    k_opt = k_opt1s[index]
    return k_opt


def opt_local_k_d5(x_tra, y_tra, m, p1, p2, d, j):
    cv_compares, k_opt1s = [], []
    b = mblocks(x_tra, y_tra, m)
    b_x = np.array(b.loc[j]['block_x'])
    b_y = np.array(b.loc[j]['block_y'])
    for i in range(5):
        cv_p1 = cv_global_k(b_x, b_y, p1, d)
        cv_p2 = cv_global_k(b_x, b_y, p2, d)
        if cv_p1 <= cv_p2:
            print('optimal k: %s, tuning at: %s'% (p1, p2-p1))
            cv_compare, k_opt1 = cv_p1, p1
            p1, p2 = p1 - int(10 / (2 ** i)), p1 + int(10 / (2 ** i))
        else:
            cv_compare, k_opt1 = cv_p2, p2
            print('optimal k: %s, tuning at: %s'% (p2, p2-p1))
            p1, p2 = p2 - int(10 / (2 ** i)), p2 + int(10 / (2 ** i))
        # print('p1,p2:', p1, p2)
        cv_compares.append(cv_compare)
        k_opt1s.append(k_opt1)
        # print('cv_compares', cv_compares)
        # print('k_opt1s', k_opt1s)
    index = cv_compares.index(min(cv_compares))
    k_opt = k_opt1s[index]
    return k_opt


# opt_k for each block in a certain fen_num / division, remember to replace: d=1, d=5
def local_parameter_k_d(x_tra, y_tra, m_knn, p1, p2, d):
    k_adapt_blocks = []
    for j in range(m_knn):
        k0 = opt_local_k_d5(x_tra, y_tra, m_knn, p1, p2, d, j)           # replace: opt_local_k_d1 or opt_local_k_d5 !!
        # print('when m = %s, block = %s, k_adapt = %s' % (m_knn, j, a))
        k_adapt_blocks.append(k0)
    print('k_adapt of the last block = %s when m_knn= %s' % (k0, m_knn))
    return k_adapt_blocks


# training parameter k for each blocks
def local_parameter_k_d_test(x_tra, y_tra, p1, p2, d, fen_num, gap):
    local_ks = {}                     # Store the opt k of each block for all fen_num
    for i in range(fen_num):
        m_knn = (i + 1) * gap
        print('m_knn:', m_knn)
        local_k = local_parameter_k_d(x_tra, y_tra, m_knn, p1, p2, d)
        local_k = np.array(local_k)
        local_k = np.squeeze(local_k)  # list object is not callable
        local_ks[i] = local_k          # Store the opt k of each block for fen_num = i
    return local_ks


'''
4. function of local average regression (LAR): NWK(gaussian),NWK(naive), NWK(Epanechnikov)
'''
'''
4.1 
when weight function is gaussian kernel, s=1,2,3
GE_gaussiank: global estimate
LE_gaussiank: the optimal local estimate
AE_gaussiank: average mixture(AVM) of all local estimates, with optimal parameter given by global 
AE_adapt_gaussiank: average mixture(AVM) of all local estimates, with adaptive optimal parameter
AE_active_gaussiank: average mixture(AVM) of all local estimates, with adaptive optimal parameter and active rule
 '''
# GE
def GE_gaussiank(x_tra, y_tra, x_tes, y_tes, h, d, s):
    n = x_tra.shape[0]
    t = x_tes.shape[0]

    # transform (n,d) to (t,d,n)
    x_tra = x_tra.T
    x_tra = np.reshape(x_tra, (1, d, n), order='A')
    x_tra = np.repeat(x_tra, t, axis=0)
    # transform (t,d) to (t,d,n)
    x_tes = np.reshape(x_tes, (t, d, 1))
    x_tes = np.repeat(x_tes, n, axis=2)

    dis = (x_tes - x_tra) / h                               # (t, d, n)
    dis_norm = (np.linalg.norm(dis, axis=1)) ** s           # (t, n)
    weight = np.exp((-dis_norm))                            # (t, n)
    a0 = np.dot(weight, y_tra)                              # (t, 1)
    b0 = np.sum(weight, axis=1)                             # (t, )
    fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)  # 0/0 = 0

    average_error = np.sum((fit - y_tes) ** 2) / t
    return average_error


# LE
def LE_gaussiank(x_tra, y_tra, x_tes1, y_tes, m, h, d, s):
    b = mblocks(x_tra, y_tra, m)
    blocks_error = []
    for i in range(m):
        b_x = np.array(b.loc[i]['block_x'])
        b_y = np.array(b.loc[i]['block_y'])
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # gaussian kernel
        dis = (x_tes - b_x) / h
        dis_norm = (np.linalg.norm(dis, axis=1)) ** s
        weight = np.exp((-dis_norm))
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)  # 0/0 = 0
        fit = np.reshape(fit, (-1, 1))
        y_tes = np.reshape(y_tes, (-1, 1))

        error = np.sum((fit - y_tes) ** 2) / t
        blocks_error.append(error)

    return min(blocks_error)


def LE_gaussiank_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, h, d, s):
    LEs_gau = []
    for i in range(fen_num):
        m_nwk = (i + 1) * gap    # fen_num = 0,1...,69, gap = 5, m_nwk = 5,10, ..., 350
        test_error = LE_gaussiank(x_tra, y_tra, x_tes, y_tes, m_nwk, h, d, s)
        LEs_gau.append(test_error)
        print('m_nwk: %s, LE_gau: %s' % (m_nwk, test_error))
    return LEs_gau


# AE
def AE_gaussiank(x_tra, y_tra, x_tes1, y_tes, m, h, d, s):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # gaussian kernel
        dis = (x_tes - b_x) / h
        dis_norm = (np.linalg.norm(dis, axis=1)) ** s
        weight = np.exp((-dis_norm))
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)  # 0/0 = 0
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)

    fit_global = np.sum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_gaussiank_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, h, d, s):
    AEs_gau = []
    for i in range(fen_num):
        m_nwk = (i + 1) * gap
        test_error = AE_gaussiank(x_tra, y_tra, x_tes, y_tes, m_nwk, h, d, s)
        AEs_gau.append(test_error)
        print('m_nwk: %s, AE_gau: %s' % (m_nwk, test_error))
    return AEs_gau


# AE_adapt
def AE_adapt_gaussiank(x_tra, y_tra, x_tes1, y_tes, m, h_adapt, d, s):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # "adaptive h" or "log h", choose one
        # h_adapt[j]: adaptive parameter "h" of jth block in ith fen_num
        # h = h_adapt[j]
        LOG = math.log(len(x_tra), n)
        h = h_adapt[j] ** LOG
        h = min(1, h)

        # gaussian kernel
        dis = (x_tes - b_x) / h
        # print('when m = %s, h_adapt for block %s is %s' %(m, j+1, h_adapt[j]))
        dis_norm = (np.linalg.norm(dis, axis=1)) ** s
        weight = np.exp((-dis_norm))
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)  # 0/0 = 0

        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)

    fit_global = np.nansum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_adapt_gaussiank_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, s, h_adapt_blocks):
    AEs_adapt_gau = []
    for i in range(fen_num):
        m_nwk = (i + 1) * gap
        h_adapt = h_adapt_blocks[i]
        test_error = AE_adapt_gaussiank(x_tra, y_tra, x_tes, y_tes, m_nwk, h_adapt, d, s)
        AEs_adapt_gau.append(test_error)
        print('when m_nwk = %s, test error = %s' % (m_nwk, test_error))
        print('-----------------------------------------------------------------------------')
    return AEs_adapt_gau


# AE_active
def AE_active_gaussiank(x_tra, y_tra, x_tes1, y_tes, m, h_adapt, d, s):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    active_m = np.zeros(n1)
    min_active, local_errors = [], []

    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # h = h_adapt[j]
        LOG = math.log(len(x_tra), n)   # log_Dj^D
        h = h_adapt[j] ** LOG
        h = min(1, h)

        # gaussian kernel
        dis = (x_tes - b_x) / h
        dis_norm = (np.linalg.norm(dis, axis=1)) ** s
        weight = np.exp((-dis_norm))
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)  # 0/0 = 0
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)

        # adopt active rule to filter out invaluable local estimates
        # active = 1 means this block is activated
        active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
        active = np.squeeze(active)
        active_m += active

        fit = fit * active.reshape(n1, 1)
        fit_blocks[:, j] = np.squeeze(fit)
        y_tes1 = np.reshape(y_tes, (-1, 1))               # (1000, ) --> (1000, 1)
        local_error = (np.sum((fit - y_tes1) ** 2)) / n1  # for adapt local estimate, to prove AVM is necessary
        local_errors.append(local_error)

    min_active.append(min(active_m))
    a1 = np.sum(fit_blocks, axis=1)
    b1 = active_m
    fit_global = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)

    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    test_inactive_index = (np.where(active_m == 0))[0]
    # print('index of test samples without active block:', test_inactive_index)
    return average_error, test_inactive_index, active_m, min_active, min(local_errors)


def AE_active_gaussiank_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, s, h_adapt_blocks):
    AEs_active_gau, min_actives, Local_errors = [], [], []
    for i in range(fen_num):
        m_nwk = (1 + i) * gap
        h_adapt = h_adapt_blocks[i]

        a = AE_active_gaussiank(x_tra, y_tra, x_tes, y_tes, m_nwk, h_adapt, d, s)
        test_error, active_num = a[0], a[2]

        AEs_active_gau.append(test_error)
        min_actives.append(a[3])
        Local_errors.append(a[4])
        print('when m_nwk = %s, test error = %s' % (m_nwk, test_error))
        print('-----------------------------------------------------------------------------')
    return AEs_active_gau, active_num, min_actives, Local_errors



'''
4.2 when weight function is naive kernel
GE_naivek, LE_naivek, AE_naivek, AE_adapt_naivek, AE_active_naivek
'''
# GE
def GE_naivek(x_tra, y_tra, x_tes, y_tes, h, d):
    t = x_tes.shape[0]
    n = x_tra.shape[0]

    # transform (n,d) to (t,d,n)
    x_tra = x_tra.T
    x_tra = np.reshape(x_tra, (1, d, n), order='A')
    x_tra = np.repeat(x_tra, t, axis=0)

    # transform (t,d) to (t,d,n)
    x_tes = np.reshape(x_tes, (t, d, 1))
    x_tes = np.repeat(x_tes, n, axis=2)

    dis = (x_tes - x_tra) / h                   # (t,d,n)
    dis_norm = (np.linalg.norm(dis, axis=1))    # (t,n)
    dis_norm = np.squeeze(dis_norm)             # (t,n)
    weight = np.where(dis_norm <= 1, 1, 0)      # weight = 1 if dis_norm <= 1
    a0 = np.dot(weight, y_tra)
    b0 = np.sum(weight, axis=1)
    fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)  # 0/0 = 0

    average_error = np.sum((fit - y_tes) ** 2) / t
    return average_error


# LE
def LE_naivek(x_tra, y_tra, x_tes1, y_tes, m, h, d):
    b = mblocks(x_tra, y_tra, m)
    blocks_error = []
    for i in range(m):
        b_x = np.array(b.loc[i]['block_x'])
        b_y = np.array(b.loc[i]['block_y'])
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # naive kernel
        dis = (x_tes - b_x) / h
        dis_norm = np.linalg.norm(dis, axis=1)
        weight = np.where(dis_norm <= 1, 1, 0)   # weight = 1 if dis_norm <= 0
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
        fit = np.reshape(fit, (-1, 1))
        y_tes = np.reshape(y_tes, (-1, 1))
        error = np.sum((fit - y_tes) ** 2) / t
        blocks_error.append(error)

    return min(blocks_error)


def LE_naivek_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, h, d):
    LEs_naive = []
    for i in range(fen_num):
        m_nwk = (i + 1) * gap
        test_error = LE_naivek(x_tra, y_tra, x_tes, y_tes, m_nwk, h, d)
        LEs_naive.append(test_error)
        print('m_nwk: %s, LE_naive: %s' % (m_nwk, test_error))
    return LEs_naive


# AE
def AE_naivek(x_tra, y_tra, x_tes1, y_tes, m, h, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # naive kernel
        dis = (x_tes - b_x) / h
        dis_norm = np.linalg.norm(dis, axis=1)
        weight = np.where(dis_norm <= 1, 1, 0)
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)

    fit_global = np.sum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_naivek_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, h, d):
    AEs_naive = []
    for i in range(fen_num):
        m_nwk = (i + 1) * gap
        test_error = AE_naivek(x_tra, y_tra, x_tes, y_tes, m_nwk, h, d)
        AEs_naive.append(test_error)
        print('m_nwk: %s, AE_naive: %s' % (m_nwk, test_error))
    return AEs_naive


# AE_adapt
def AE_adapt_naivek(x_tra, y_tra, x_tes1, y_tes, m, h_adapt, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # h = h_adapt[j]
        LOG = math.log(len(x_tra), n)   # log_Dj^D
        h = h_adapt[j] ** LOG
        h = min(1, h)

        # naive kernel
        dis = (x_tes - b_x) / h
        dis_norm = np.linalg.norm(dis, axis=1)
        weight = np.where(dis_norm <= 1, 1, 0)
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)

    fit_global = np.nansum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_adapt_naivek_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, h_adapt_blocks):
    AEs_adapt_naive = []
    for i in range(fen_num):
        m_nwk = (i + 1) * gap
        h_adapt = h_adapt_blocks[i]
        test_error = AE_adapt_naivek(x_tra, y_tra, x_tes, y_tes, m_nwk, h_adapt, d)
        AEs_adapt_naive.append(test_error)
        print('when m_nwk = %s, test error = %s' % (m_nwk, test_error))
        print('-----------------------------------------------------------------------------')
    return AEs_adapt_naive


# AE_active
def AE_active_naivek(x_tra, y_tra, x_tes1, y_tes, m, h_adapt, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    active_m = np.zeros(n1)
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # h = h_adapt[j]
        LOG = math.log(len(x_tra), n)
        h = h_adapt[j] ** LOG
        h = min(1, h)

        # naive kernel
        dis = (x_tes - b_x) / h
        dis_norm = np.linalg.norm(dis, axis=1)
        weight = np.where(dis_norm <= 1, 1, 0)
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)  # 0/0 = 0
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)

        # qualifing
        active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
        active = np.squeeze(active)
        active_m += active
        fit = fit * active.reshape(n1, 1)
        fit_blocks[:, j] = np.squeeze(fit)  # m * (n, )  --> (n, m)

    # print('active_m', active_m)
    # active_m
    a1 = np.sum(fit_blocks, axis=1)
    b1 = active_m
    fit_global = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)  # 0/0 = 0
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    test_inactive_index = (np.where(active_m == 0))[0]  # index of test samples without active block
    return average_error, test_inactive_index


def AE_active_naivek_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, h_adapt_blocks):
    AEs_active_nai = []
    for i in range(fen_num):
        m_nwk = (i + 1) * gap
        h_adapt = h_adapt_blocks[i]
        test_error = AE_active_naivek(x_tra, y_tra, x_tes, y_tes, m_nwk, h_adapt, d)[0]
        AEs_active_nai.append(test_error)
        print('when m_nwk = %s, test error = %s' % (m_nwk, test_error))
        print('-----------------------------------------------------------------------------')
    return AEs_active_nai


'''
4.3 when weight function is Epanechnikov kernel
GE_Epank, LE_Epank, AE_Epank, AE_adapt_Epank, AE_active_Epank
'''
# GE
def GE_Epank(x_tra, y_tra, x_tes, y_tes, h, d):
    t = x_tes.shape[0]
    n = x_tra.shape[0]

    # transform (n,d) to (t,d,n)
    x_tra = x_tra.T
    x_tra = np.reshape(x_tra, (1, d, n), order='A')
    x_tra = np.repeat(x_tra, t, axis=0)
    # transform (t,d) to (t,d,n)
    x_tes = np.reshape(x_tes, (t, d, 1))
    x_tes = np.repeat(x_tes, n, axis=2)

    dis = (x_tes - x_tra) / h
    dis_norm = (np.linalg.norm(dis, axis=1)) ** 2
    dis_norm = np.squeeze(dis_norm)
    weight = np.where((1 - dis_norm) > 0, 1 - dis_norm, 0)
    a0 = np.dot(weight, y_tra)
    b0 = np.sum(weight, axis=1)
    fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)  # 0/0 = 0

    average_error = np.sum((fit - y_tes) ** 2) / t
    return average_error


# LE
def LE_Epank(x_tra, y_tra, x_tes1, y_tes, m, h, d):
    b = mblocks(x_tra, y_tra, m)
    blocks_error = []
    for i in range(m):
        b_x = np.array(b.loc[i]['block_x'])
        b_y = np.array(b.loc[i]['block_y'])
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # Epan kernel
        dis = (x_tes - b_x) / h
        dis_norm = (np.linalg.norm(dis, axis=1)) ** 2
        weight = np.where((1 - dis_norm) > 0, 1 - dis_norm, 0)
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
        fit = np.reshape(fit, (-1, 1))
        y_tes = np.reshape(y_tes, (-1, 1))
        error = np.sum((fit - y_tes) ** 2) / t
        blocks_error.append(error)
    return min(blocks_error)


def LE_Epank_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, h, d):
    LEs_epan = []
    for i in range(fen_num):
        m_nwk = (i + 1) * gap
        test_error = LE_Epank(x_tra, y_tra, x_tes, y_tes, m_nwk, h, d)
        LEs_epan.append(test_error)
        print('m_nwk: %s, LE_epan: %s' % (m_nwk, test_error))
    return LEs_epan


# AE
def AE_Epank(x_tra, y_tra, x_tes1, y_tes, m, h, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # Epan kernel
        dis = (x_tes - b_x) / h
        dis_norm = (np.linalg.norm(dis, axis=1)) ** 2
        weight = np.where((1 - dis_norm) > 0, 1 - dis_norm, 0)
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
        fit = np.reshape(fit, (-1, 1))  # reshape (t, ) to (t, 1)
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)

    fit_global = np.sum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_Epank_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, h, d):
    AEs_nwk = []
    for i in range(fen_num):
        m_nwk = (i + 1) * gap
        test_error = AE_Epank(x_tra, y_tra, x_tes, y_tes, m_nwk, h, d)
        AEs_nwk.append(test_error)
        print('m_nwk: %s, AE_epan: %s' % (m_nwk, test_error))
    return AEs_nwk


# AE_adapt
def AE_adapt_epank(x_tra, y_tra, x_tes1, y_tes, m, h_adapt, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        #h = h_adapt[j]
        LOG = math.log(len(x_tra), n)
        h = h_adapt[j] ** LOG
        h = min(1, h)

        # Epan kernel
        dis = (x_tes - b_x) / h
        # print('when m = %s, h_adapt for block %s is %s' %(m, j+1, h_adapt[j]))
        dis_norm = (np.linalg.norm(dis, axis=1)) ** 2
        weight = np.where((1 - dis_norm) > 0, 1 - dis_norm, 0)
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)

    fit_global = np.nansum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_adapt_epank_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, h_blocks_adapt):
    AEs_adapt_epan = []
    for i in range(fen_num):
        m_nwk = (i + 1) * gap
        h_adapt = h_blocks_adapt[i]
        test_error = AE_adapt_epank(x_tra, y_tra, x_tes, y_tes, m_nwk, h_adapt, d)
        AEs_adapt_epan.append(test_error)
        print('when m_nwk = %s, test error = %s' % (m_nwk, test_error))
        print('-----------------------------------------------------------------------------')
    return AEs_adapt_epan


# AE_active
def AE_active_epank(x_tra, y_tra, x_tes1, y_tes, m, h_adapt, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    active_m = np.zeros(n1)
    min_active, local_errors = [], []

    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        #h = h_adapt[j]
        LOG = math.log(len(x_tra), n)
        h = h_adapt[j] ** LOG
        h = min(1, h)

        # Epan kernel
        dis = (x_tes - b_x) / h
        dis_norm = (np.linalg.norm(dis, axis=1)) ** 2
        weight = np.where((1 - dis_norm) > 0, 1 - dis_norm, 0)
        a0 = np.dot(weight, b_y)
        b0 = np.sum(weight, axis=1)
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)  # 0/0 = 0
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)

        # qualifing
        active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)  # new active rule: block is activated if fit >= 1/b_x_n
        active = np.squeeze(active)
        active_m += active
        fit = fit * active.reshape(n1, 1)
        fit_blocks[:, j] = np.squeeze(fit)

        y_tes1 = np.reshape(y_tes, (-1, 1))
        local_error = (np.sum((fit - y_tes1) ** 2)) / n1
        local_errors.append(local_error)

    min_active.append(min(active_m))
    a1 = np.sum(fit_blocks, axis=1)
    b1 = active_m
    fit_global = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)

    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    test_inactive_index = (np.where(active_m == 0))[0]
    # print('index of test sample without active block:', test_inactive_index)
    return average_error, test_inactive_index, active_m, min_active, min(local_errors)


def AE_active_epank_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, h_blocks_adapt):
    AEs_active_epan, min_actives, Local_errors = [], [], []
    for i in range(fen_num):
        m_nwk = (1 + i) * gap
        h_adapt = h_blocks_adapt[i]
        a = AE_active_epank(x_tra, y_tra, x_tes, y_tes, m_nwk, h_adapt, d)
        test_error, active_num = a[0], a[2]
        AEs_active_epan.append(test_error)
        min_actives.append(a[3])
        Local_errors.append(a[4])
        print('when m_nwk = %s, test error = %s' % (m_nwk, test_error))
        print('-----------------------------------------------------------------------------')
    return AEs_active_epan, active_num, min_actives, Local_errors


'''
4.4 when Knn
GE_Knn, LE_Knn, AE_Knn, AE_adapt_Knn, AE_active_Knn
'''
# GE
def GE_Knn(x_tra, y_tra, x_tes, y_tes, k, d):
    t = x_tes.shape[0]
    n = x_tra.shape[0]
    # transform (n,d) to (t,d,n)
    x_tra = x_tra.T
    x_tra = np.reshape(x_tra, (1, d, n), order='A')
    x_tra = np.repeat(x_tra, t, axis=0)

    # transform (t,d) to (t,d,n)
    x_tes = np.reshape(x_tes, (t, d, 1))
    x_tes = np.repeat(x_tes, n, axis=2)

    dis = (x_tes - x_tra)
    dis_norm = np.linalg.norm(dis, axis=1)
    index = np.argsort(dis_norm, axis=1)
    y_knn = y_tra[index]
    y_knn = y_knn[:, :k]
    #fit = (np.sum(y_knn, -1)) / k
    a0 = np.sum(y_knn, -1)
    b0 = k
    fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)

    average_error = np.sum((fit - y_tes) ** 2) / t
    return average_error


# LE
def LE_Knn(x_tra, y_tra, x_tes1, y_tes, m, k, d):
    b = mblocks(x_tra, y_tra, m)
    blocks_error = []
    b_k = int(k / m)  # b_k = math.ceil(k / m)
    # print(b_k)
    for i in range(m):
        b_x = np.array(b.loc[i]['block_x'])
        b_y = np.array(b.loc[i]['block_y'])
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # Knn
        dis = (x_tes - b_x)
        dis_norm = np.linalg.norm(dis, axis=1)
        index = np.argsort(dis_norm, axis=1)
        y_knn = b_y[index]
        y_knn = y_knn[:, :b_k]

        a0 = np.sum(y_knn, -1)
        b0 = b_k
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
        fit = np.reshape(fit, (-1, 1))
        y_tes = np.reshape(y_tes, (-1, 1))
        error = np.sum((fit - y_tes) ** 2) / t
        blocks_error.append(error)
    return min(blocks_error)


def LE_Knn_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, k, d):
    LEs_knn = []
    for i in range(fen_num):
        m_knn = (i + 1) * gap
        test_error = LE_Knn(x_tra, y_tra, x_tes, y_tes, m_knn, k, d)
        LEs_knn.append(test_error)
        print('m_knn: %s, LE_knn: %s' % (m_knn, test_error))
    return LEs_knn


# AE
def AE_Knn(x_tra, y_tra, x_tes1, y_tes, m, k, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    b_k = int(k / m)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # Knn
        dis = (x_tes - b_x)
        dis_norm = np.linalg.norm(dis, axis=1)
        index = np.argsort(dis_norm, axis=1)  # axis=i means sorting based on dimension i
        y_knn = b_y[index]                    # sorted b_y
        y_knn = y_knn[:, :b_k]

        a0 = np.sum(y_knn, -1)
        b0 = b_k
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
        fit_blocks[:, j] = np.squeeze(fit)

    fit_global = np.sum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_Knn_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, k, d):
    AEs_knn = []
    for i in range(fen_num):
        m_knn = (i + 1) * gap
        test_error = AE_Knn(x_tra, y_tra, x_tes, y_tes, m_knn, k, d)
        AEs_knn.append(test_error)
        print('m_knn: %s, AE_knn: %s' % (m_knn, test_error))
    return AEs_knn


# AE_adapt
def AE_adapt_Knn(x_tra, y_tra, x_tes1, y_tes, m, k_adapt, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # k = k_adapt[j]
        LOG = math.log(len(x_tra), n)
        k = (k_adapt[j] ** LOG) / m        # <class 'numpy.float64'>
        k = math.ceil(k)

        # Knn
        dis = (x_tes - b_x)
        dis_norm = np.linalg.norm(dis, axis=1)
        index = np.argsort(dis_norm, axis=1)
        y_knn = b_y[index]
        y_knn = y_knn[:, :k]
        a0 = np.sum(y_knn, -1)
        b0 = k
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)
        fit_blocks = np.nan_to_num(fit_blocks)

    fit_global = np.sum(fit_blocks, axis=1) / m
    error = (fit_global - y_tes) ** 2
    average_error = np.sum(error) / n1
    return average_error


def AE_adapt_Knn_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, k_blocks_adapt):
    AEs_knnadapt = []
    for i in range(fen_num):
        m_knn = (i + 1) * gap
        k_adapt = k_blocks_adapt[i]
        test_error = AE_adapt_Knn(x_tra, y_tra, x_tes, y_tes, m_knn, k_adapt, d)
        AEs_knnadapt.append(test_error)
        print('m_knn: %s, AE_Knn: %s' % (m_knn, test_error))
    return AEs_knnadapt


# AE_active
def AE_active_Knn(x_tra, y_tra, x_tes1, y_tes, m, k_adapt, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    active_m = np.zeros(n1)
    min_active, local_errors = [], []
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]

        # transform: (t, d, n)
        x_tes = np.reshape(x_tes, (t, d, 1))
        x_tes = np.repeat(x_tes, n, axis=2)
        b_x = b_x.T
        b_x = np.reshape(b_x, (1, d, n), order='A')
        b_x = np.repeat(b_x, t, axis=0)

        # k = k_adapt[j]
        LOG = math.log(len(x_tra), n)
        k = (k_adapt[j] ** LOG) / m
        k = math.ceil(k)

        # Knn
        dis = (x_tes - b_x)
        dis_norm = np.linalg.norm(dis, axis=1)
        index = np.argsort(dis_norm, axis=1)
        y_knn = b_y[index]
        y_knn = y_knn[:, :k]
        a0 = np.sum(y_knn, -1)
        b0 = k
        fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)

        # qualifing
        active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
        active = np.squeeze(active)
        active_m += active
        fit = fit * (active.reshape(n1, 1))
        fit_blocks[:, j] = np.squeeze(fit)
        # fit_blocks = np.nan_to_num(fit_blocks)
        y_tes1 = np.reshape(y_tes, (-1, 1))
        local_error = (np.sum((fit - y_tes1) ** 2)) / n1
        local_errors.append(local_error)

    min_active.append(min(active_m))
    a1 = np.sum(fit_blocks, axis=1)
    b1 = active_m
    fit_global = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)  # 0/0 = 0
    average_error = np.sum((fit_global - y_tes) ** 2 ) / n1
    return average_error, min(local_errors)


def AE_active_Knn_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, k_blocks_adapt):
    AEs_active_knn, Local_errors = [], []
    for i in range(fen_num):
        m_knn = (i + 1) * gap
        k_adapt = k_blocks_adapt[i]
        test_error = AE_active_Knn(x_tra, y_tra, x_tes, y_tes, m_knn, k_adapt, d)
        AEs_active_knn.append(test_error[0])
        Local_errors.append(test_error[1])
        print('m_knn: %s, AE_knn: %s' % (m_knn, test_error[0]))
    return AEs_active_knn, Local_errors


'''
4.5 hybrid algorithm
20 trails use the same random order, because our aim is to prove the stability of the algorithm to the "data", 
regardless of the previous np.random.seed(i), random.randint (1, 4) are all the same sequence.

(1) "assign = random.randint(1, 4)":  to ensure the autonomy on algorithm
     assign=1, NWK(gaussian); assign=2, NWK(Epanechnikov); assign=3, NWK(naive); assign=4, knn
(2) if hybrid_algorithm on 3 algorithms, change random.randint(1, 4) to random.randint(1, 3); and change adapt_
'''
# def hybrid_algorithm(x_tra, y_tra, x_tes1, y_tes, m, fen_now, adapt_gau, adapt_epan, adapt_nai, adapt_knn, d):
#     b = mblocks(x_tra, y_tra, m)
#     n1 = len(x_tes1)
#     fit_blocks = np.zeros((n1, m))
#     active_m = np.zeros(n1)
#     random.seed(0)
#     for i in range(m):
#         assign = random.randint(1, 4)
#         b_x = np.array(b.loc[i]['block_x'])
#         b_y = np.array(b.loc[i]['block_y'])
#         x_tes = np.array(x_tes1)              # ensure x_tes.shape: (t, d) each time
#         n = b_x.shape[0]
#         t = x_tes.shape[0]
#
#         # transform: (t, d, n)
#         x_tes = np.reshape(x_tes, (t, d, 1))
#         x_tes = np.repeat(x_tes, n, axis=2)
#         b_x = b_x.T
#         b_x = np.reshape(b_x, (1, d, n), order='A')
#         b_x = np.repeat(b_x, t, axis=0)
#         LOG = math.log(len(x_tra), n)         # logarithmic operator
#
#         # gaussian kernel, s=2
#         if assign == 1:
#             #print('gau')
#             h = adapt_gau[fen_now][i] ** LOG  # adapt local h of block[i] at a certain fen_num[fen_num]
#             h = min(1, h)
#             # print(adapt_g2[fen_now][i])
#
#             # gau_s2
#             dis = (x_tes - b_x) / h
#             dis_norm = (np.linalg.norm(dis, axis=1)) ** 2
#             weight = np.exp((-dis_norm))
#             a0 = np.dot(weight, b_y)
#             b0 = np.sum(weight, axis=1)
#             fit = np.divide(a0, b0, out=np.zeros_like(a0, dtype=np.float64), where=b0 != 0)
#             fit = np.reshape(fit, (-1, 1))
#             fit = np.nan_to_num(fit)
#
#             # qualifing
#             active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
#             active = np.squeeze(active)
#             active_m += active
#             fit = fit * active.reshape(n1, 1)
#             fit_blocks[:, i] = np.squeeze(fit)
#
#         # Epan kernel
#         if assign == 2:
#             #print('epan')
#             h = adapt_epan[fen_now][i] ** LOG
#             h = min(1, h)
#             # print(adapt_epan[fen_now][i])
#
#             # epan kernel
#             dis = (x_tes - b_x) / h
#             dis_norm = (np.linalg.norm(dis, axis=1)) ** 2
#             weight = np.where((1 - dis_norm) > 0, 1 - dis_norm, 0)
#             a1 = np.dot(weight, b_y)
#             b1 = np.sum(weight, axis=1)
#             fit = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)
#             fit = np.reshape(fit, (-1, 1))
#             fit = np.nan_to_num(fit)
#
#             # qualifing
#             active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
#             active = np.squeeze(active)
#             active_m += active
#             fit = fit * active.reshape(n1, 1)
#             fit_blocks[:, i] = np.squeeze(fit)
#
#         # naive kernel
#         if assign == 3:
#             #print('naive')
#             h = adapt_nai[fen_now][i] ** LOG
#             h = min(1, h)
#             # print(adapt_nai[fen_now][i])
#
#             # naive kernel
#             dis = (x_tes - b_x) / h
#             dis_norm = np.linalg.norm(dis, axis=1)
#             weight = np.where(dis_norm <= 1, 1, 0)
#             a2 = np.dot(weight, b_y)
#             b2 = np.sum(weight, axis=1)
#             fit = np.divide(a2, b2, out=np.zeros_like(a2, dtype=np.float64), where=b2 != 0)
#             fit = np.reshape(fit, (-1, 1))
#             fit = np.nan_to_num(fit)
#
#             # qualifing
#             active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
#             active = np.squeeze(active)
#             active_m += active
#             fit = fit * active.reshape(n1, 1)
#             fit_blocks[:, i] = np.squeeze(fit)
#
#         # knn
#         if assign == 4:
#             #print('knn')
#             k = ((adapt_knn[fen_now][i]) ** LOG) / m
#             k = math.ceil(k)
#             # print((adapt_knn[fen_now][i]))
#
#             # knn estimate
#             dis = (x_tes - b_x)
#             dis_norm = np.linalg.norm(dis, axis=1)
#             index = np.argsort(dis_norm, axis=1)
#             y_knn = b_y[index]
#             y_knn = y_knn[:, :k]
#             fit = (np.sum(y_knn, -1)) / k
#             fit = np.reshape(fit, (-1, 1))
#             fit = np.nan_to_num(fit)
#
#             # qualifing
#             active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
#             active = np.squeeze(active)
#             active_m += active
#             fit = fit * (active.reshape(n1, 1))
#             fit_blocks[:, i] = np.squeeze(fit)
#             # fit_blocks = np.nan_to_num(fit_blocks)
#
#     # print('active m after knn', active_m[:5])
#     a3 = np.sum(fit_blocks, axis=1)
#     b3 = active_m
#     fit_global = np.divide(a3, b3, out=np.zeros_like(a3, dtype=np.float64), where=b3 != 0)
#     average_error = np.sum((fit_global - y_tes) ** 2) / n1
#     return average_error
#
#
# def hybrid_algorithm_test(x_tra, y_tra, x_tes, y_tes, adapt_gau, adapt_epan, adapt_nai, adapt_knn, d, gap, fen_num):
#     Hybrid_Error= []
#     for i in range(fen_num-1, fen_num):
#         m = (i + 1) * gap
#         fen_now = i
#         print('fen_now: %s, m:%s' % (fen_now, m))
#         hybrid_error = hybrid_algorithm(x_tra, y_tra, x_tes, y_tes, m, fen_now, adapt_gau, adapt_epan, adapt_nai, adapt_knn, d)
#         Hybrid_Error.append(hybrid_error)
#         print('Hybrid_Error:', Hybrid_Error)
#     return Hybrid_Error



'''
5. Appendix: Krr, AdaBoost, RF ----------------------------
'''
'''
5.1 Krr ---------------------------------------------------  
'''
# kernel function when d=1
def kernel_d1(x,y):
    n,t = len(x), len(y)
    mat_1 = np.ones((t,n))
    y_1 = np.reshape(y, (t, 1), order='A')  # (t, 1)
    y_1 = np.repeat(y_1, n, axis=1)         # (t, n)
    x_1 = np.reshape(x, (1, n), order='A')  # (1, n)
    x_1 = np.repeat(x_1, t, axis=0)         # (t, n)
    kermatrix = np.minimum(x_1, y_1) + mat_1
    return kermatrix


# kernel function when d=5
def function_h5(x):
    if x >= 0 and x <= 1:
        result = (1 - x) ** 4 * (4*x+1)  # 0.0
    else:
        result = 0.0
    return result


def kernel_d3(x,y):
    n, t = len(x), len(y)
    y_1 = np.reshape(y, (t, 1, 5))
    y_1 = np.repeat(y_1, n, axis=1)  # (t,n,5)
    x_1 = np.reshape(x, (1, n, 5))
    x_1 = np.repeat(x_1, t, axis=0)  # (t,n,5)
    dis = x_1 - y_1
    dis_norm = np.linalg.norm(dis, axis=2)   # (t,n)
    # kernel function（1）：h5
    function_h5_vector = np.vectorize(function_h5)
    kermatrix = function_h5_vector(dis_norm)
    # kernel function（2）：fai
    # function_fai_vector = np.vectorize(function_fai)
    # kermatrix = function_fai_vector(dis_norm)
    return kermatrix


# calculate parameter alpha
def Alpha_KRR(X_tra, y_tra, d, lambda_krr):
    n = len(X_tra)
    ker = kernel_d1(X_tra, X_tra) if d == 1 else kernel_d3(X_tra, X_tra)
    U, S, V = np.linalg.svd(ker)
    # print(np.min(S))
    # print(len(S))

    S_lambda = S + n * lambda_krr
    condition_n = np.max(S_lambda) / np.min(S_lambda)

    S_lambda = 1 / S_lambda  # (n, )
    S_lambda = np.reshape(S_lambda, (1, n), order='A')  # (1, n)
    S_lambda = np.repeat(S_lambda, n, axis=0)           # (n, n)
    US = np.multiply(U, S_lambda)
    ker_inverse = np.dot(US, V)
    alpha = np.dot(ker_inverse, np.reshape(y_tra, (len(y_tra), 1)))
    return alpha, condition_n


# calculate MSE
def Predictedkrr(x_tra, y_tra, x_tes, y_tes, d, regular_para):
    n, t = len(x_tra), len(x_tes)
    pred = Alpha_KRR(x_tra, y_tra, d, regular_para)
    pred_alpha, pred_condition = pred[0], pred[1]
    if d == 1:
        pred_ker = kernel_d1(x_tra, x_tes)
        y_fit = np.dot(pred_ker, pred_alpha)
        y_fit = np.squeeze(y_fit)
    else:
        pred_ker = kernel_d3(x_tra, x_tes)
        y_fit = np.dot(pred_ker, pred_alpha)
        y_fit = np.squeeze(y_fit)
    average_error = np.sum((y_fit - y_tes) ** 2) / t
    return y_fit, average_error, pred_condition


# calculate regularization parameter
def cv_lambda_krr(x, y, f, d, lambda_krr):
    kf = KFold(n_splits=f, shuffle=True)
    # kf = KFold(n_splits=f)
    error = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        result = Predictedkrr(x_train, y_train, x_test, y_test, d, lambda_krr)
        error.append(result[1])
    return sum(error)/f


def global_parameter_lambda_krr(x_tra, y_tra, f, d):
    cv_errors, lambda_opts = [], []
    for i in range(10):  # d=1
        lambda_i = 0.00001 +0.0005*i
        print('lambda_i:', lambda_i)
        cv_error = cv_lambda_krr(x_tra, y_tra, f, d, lambda_i)
        cv_errors.append(cv_error)
        lambda_opts.append(lambda_i)
    index = cv_errors.index(min(cv_errors))
    lambda_opt = lambda_opts[index]
    return lambda_opt


def parameter_lambda_krr_forplot(x_tra, y_tra, f, d):
    errors, condition_ns, lambda_opts = [], [], []
    for i in range(10):  # f1
        lambda_i = 0.00001 + 0.0005 * i
        print('lambda_i:', lambda_i)
        cv = cv_lambda_krr(x_tra, y_tra, f, d, lambda_i)
        errors.append(cv)
        lambda_opts.append(lambda_i)
    return errors, lambda_opts



# in equal-sized setting, N=2000，the number of local agent =2，4，6...,40 ------------------
# partition
def opt_local_lambda(x_tra, y_tra, m, f, d, j):
    b = mblocks(x_tra, y_tra, m)
    b_x = np.array(b.loc[j]['block_x'])
    b_y = np.array(b.loc[j]['block_y'])
    lambda_opt = global_parameter_lambda_krr(b_x, b_y, f, d)
    return lambda_opt


# opt_lambda for each block of a certain fen_num / division
def local_parameter_lambda(x_tra, y_tra, num_agent, f, d):
    lambda_adapt_blocks = []
    for j in range(num_agent):
        lambda0 = opt_local_lambda(x_tra, y_tra, num_agent, f, d, j)
        print('when num_agent = %s, block = %s, lambda_adapt = %s' % (num_agent, j, lambda0))
        lambda_adapt_blocks.append(lambda0)
    return lambda_adapt_blocks


# training parameter lambda for each blocks
def local_parameter_lambda_train(x_tra, y_tra, f, d, fen_num, gap):
    local_lambdas = {}
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        print('num_agent:', num_agent)
        local_lambda = local_parameter_lambda(x_tra, y_tra, num_agent, f, d)
        local_lambda = np.array(local_lambda)
        local_lambda = np.squeeze(local_lambda)
        local_lambdas[i] = local_lambda
    return local_lambdas



def LE_krr(x_tra, y_tra, x_tes1, y_tes, m, lambda1, d):
    b = mblocks(x_tra, y_tra, m)
    blocks_error = []
    for i in range(m):
        b_x = np.array(b.loc[i]['block_x'])
        b_y = np.array(b.loc[i]['block_y'])
        x_tes = np.array(x_tes1)  # ensure x_tes.shape: (t, d) each time
        error = Predictedkrr(b_x, b_y, x_tes, y_tes, d, lambda1)[1]  # KRR
        blocks_error.append(error)
    return min(blocks_error)


def LE_krr_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, lambda_para, d):
    LEs_krr = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        test_error = LE_krr(x_tra, y_tra, x_tes, y_tes, num_agent, lambda_para, d)
        LEs_krr.append(test_error)
        print('num_agent: %s, LEs_krr: %s' % (num_agent, test_error))
    return LEs_krr


def AE_krr_opt(x_tra, y_tra, x_tes1, y_tes, m, lambda1, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        fit = Predictedkrr(b_x, b_y, x_tes, y_tes, d, lambda1)[0]  # KRR
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)
    fit_global = np.sum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_krr_opt_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, lambda_para, d):
    AEs_krr = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        test_error = AE_krr_opt(x_tra, y_tra, x_tes, y_tes, num_agent, lambda_para, d)
        AEs_krr.append(test_error)
        print('num_agent: %s, AE_krr: %s' % (num_agent, test_error))
    return AEs_krr



def AE_adapt_krr(x_tra, y_tra, x_tes1, y_tes, m, l_lambda, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]

        # "adaptive h" or "log h", choose one
        # h_adapt[j]: adaptive parameter "h" of jth block in ith fen_num
        #h = h_adapt[j]
        LOG = math.log(len(x_tra), n)
        l_lambda_log = l_lambda[j] ** LOG

        # print('l_lambda[j]', l_lambda[j])
        # print('LOG', LOG)
        # print('l_lambda_log', l_lambda_log)

        fit = Predictedkrr(b_x, b_y, x_tes, y_tes, d, l_lambda_log)[0]  # KRR
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)

    fit_global = np.nansum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_adapt_krr_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, l_lambda_blocks):
    AEs_adapt_krr = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        l_lambda = l_lambda_blocks[i]
        test_error = AE_adapt_krr(x_tra, y_tra, x_tes, y_tes, num_agent, l_lambda, d)
        AEs_adapt_krr.append(test_error)
        print('when num_agent = %s, test error = %s' % (num_agent, test_error))
        print('-----------------------------------------------------------------------------')
    return AEs_adapt_krr


# AE_active
def AE_active_krr(x_tra, y_tra, x_tes1, y_tes, m, l_lambda, d):
    b = mblocks(x_tra, y_tra, m)    # same
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    min_active, local_errors = [], []
    active_m = np.zeros(n1)
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])  # same
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]

        LOG = math.log(len(x_tra), n)
        l_lambda_log = l_lambda[j] ** LOG

        fit = Predictedkrr(b_x, b_y, x_tes, y_tes, d, l_lambda_log)[0]  # KRR
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)

        # adopt active rule to filter out invaluable local estimates
        # active = 1 means this block is activated
        active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
        active = np.squeeze(active)
        active_m += active

        fit = fit * active.reshape(n1, 1)
        fit_blocks[:, j] = np.squeeze(fit)
        y_tes1 = np.reshape(y_tes, (-1, 1))               # (1000, ) --> (1000, 1)
        local_error = (np.sum((fit - y_tes1) ** 2)) / n1  # for adapt local estimate, to prove AVM is necessary
        local_errors.append(local_error)

    min_active.append(min(active_m))
    a1 = np.sum(fit_blocks, axis=1)
    b1 = active_m
    fit_global = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)

    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    test_inactive_index = (np.where(active_m == 0))[0]
    # print('index of test samples without active block:', test_inactive_index)
    return average_error, min(local_errors)  # lE用自己的参数，用log机制和active机制


def AE_active_krr_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, l_lambda_blocks):
    AEs_active_krr, Local_errors = [], []
    for i in range(fen_num):
        num_agent = (1 + i) * gap
        l_lambda = l_lambda_blocks[i]
        a = AE_active_krr(x_tra, y_tra, x_tes, y_tes, num_agent, l_lambda, d)
        AEs_active_krr.append(a[0])
        Local_errors.append(a[1])
        print('when num_agent = %s, AE_log_active = %s' % (num_agent, a[0]))
        print('-----------------------------------------------------------------------------')
    return AEs_active_krr, Local_errors



# in unequal-sized setting, N=2000，the number of local agent =2，4，6...,40 ------------------
# partition
def opt_local_lambda_diff(x_tra, y_tra, m, f, d, j):
    dic = mblocks_diff(x_tra, m)
    b_x = x_tra[dic[j]]
    b_y = y_tra[dic[j]]
    print('-----', b_x.shape)
    lambda_opt = global_parameter_lambda_krr(b_x, b_y, f, d)
    return lambda_opt


# opt_lambda for each block of a certain fen_num / division
def local_parameter_lambda_diff(x_tra, y_tra, num_agent, f, d):
    lambda_adapt_blocks = []
    for j in range(num_agent):
        lambda0 = opt_local_lambda_diff(x_tra, y_tra, num_agent, f, d, j)
        print('when num_agent = %s, block = %s, lambda_adapt = %s' % (num_agent, j, lambda0))
        lambda_adapt_blocks.append(lambda0)
    return lambda_adapt_blocks


# training parameter lambda for each blocks
def local_parameter_lambda_train_diff(x_tra, y_tra, f, d, fen_num, gap):
    local_lambdas = {}
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        print('num_agent:', num_agent)
        local_lambda = local_parameter_lambda_diff(x_tra, y_tra, num_agent, f, d)
        local_lambda = np.array(local_lambda)
        local_lambda = np.squeeze(local_lambda)
        local_lambdas[i] = local_lambda
    return local_lambdas


def LE_krr_adapt(x_tra, y_tra, x_tes1, y_tes, m, l_lambda, d):
    dic = mblocks_diff(x_tra, m)
    blocks_error = []
    for j in range(m):
        b_x = x_tra[dic[j]]
        b_y = y_tra[dic[j]]
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        l_lambda_1 = l_lambda[j]
        error = Predictedkrr(b_x, b_y, x_tes, y_tes, d, l_lambda_1)[1]  # KRR
        blocks_error.append(error)
    return min(blocks_error)


def LE_krr_adapt_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, l_lambda_blocks, d):
    LEs_krr = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        l_lambda = l_lambda_blocks[i]
        test_error = LE_krr_adapt(x_tra, y_tra, x_tes, y_tes, num_agent, l_lambda, d)
        LEs_krr.append(test_error)
        print('m: %s, LE_adapt: %s' % (num_agent, test_error))
    return LEs_krr


def AE_active_krr_diff(x_tra, y_tra, x_tes1, y_tes, m, l_lambda, d):
    dic = mblocks_diff(x_tra, m)    # diff
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    min_active, local_errors = [], []
    active_m = np.zeros(n1)
    for j in range(m):
        b_x = x_tra[dic[j]]                  # diff
        b_y = y_tra[dic[j]]
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]

        LOG = math.log(len(x_tra), n)
        l_lambda_log = l_lambda[j] ** LOG

        fit = Predictedkrr(b_x, b_y, x_tes, y_tes, d, l_lambda_log)[0]  # KRR
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)

        # adopt active rule to filter out invaluable local estimates
        # active = 1 means this block is activated
        active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
        active = np.squeeze(active)
        active_m += active

        fit = fit * active.reshape(n1, 1)
        fit_blocks[:, j] = np.squeeze(fit)
        y_tes1 = np.reshape(y_tes, (-1, 1))               # (1000, ) --> (1000, 1)
        local_error = (np.sum((fit - y_tes1) ** 2)) / n1  # for adapt local estimate, to prove AVM is necessary
        local_errors.append(local_error)

    min_active.append(min(active_m))
    a1 = np.sum(fit_blocks, axis=1)
    b1 = active_m
    fit_global = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)

    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    test_inactive_index = (np.where(active_m == 0))[0]
    # print('index of test samples without active block:', test_inactive_index)
    return average_error, min(local_errors)  # lE用自己的参数，用log机制和active机制


def AE_active_krr_diff_test(x_tra, y_tra, x_tes, y_tes, d, fen_num, gap, l_lambda_blocks):
    AEs_active_krr, Local_errors = [], []
    for i in range(fen_num):
        num_agent = (1 + i) * gap
        l_lambda = l_lambda_blocks[i]
        a = AE_active_krr_diff(x_tra, y_tra, x_tes, y_tes, num_agent, l_lambda, d)
        AEs_active_krr.append(a[0])
        Local_errors.append(a[1])
        print('when num_agent = %s, AE_log_active = %s' % (num_agent, a[0]))
        print('-----------------------------------------------------------------------------')
    return AEs_active_krr, Local_errors


'''
5.2 Boosting ---------------------------------------------------  
'''
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# l_t = 2
# LOG = math.log(2000, 500) # 10.965784284662087
# l_t_log = math.ceil(l_t ** LOG)


# Decision trees implemented in sklearn are all binary trees (cart algorithm)
def Predictedboost(x_tra, y_tra, x_tes, y_tes, regular_para):
    n, t = len(x_tra), len(x_tes)
    regr_1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=regular_para, random_state=0)
    regr_1.fit(x_tra, y_tra)
    y_fit = regr_1.predict(x_tes)
    y_fit = np.squeeze(y_fit)
    average_error = np.sum((y_fit - y_tes) ** 2) / t
    return y_fit, average_error


def cv_t_boost(x, y, f, t_boost):
    kf = KFold(n_splits=f, shuffle=True)
    # kf = KFold(n_splits=f)
    error = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        result = Predictedboost(x_train, y_train, x_test, y_test, t_boost)
        error.append(result[1])
    return sum(error)/f


def global_parameter_t_boost(x_tra, y_tra, f):
    cv_errors, t_opts = [], []
    for i in range(1,31):   #  if d=5，range(1, 41), 2 *
        t_boost = int(i*2)
        # print('t_boost:', t_boost)
        cv_error = cv_t_boost(x_tra, y_tra, f, t_boost)
        cv_errors.append(cv_error)
        t_opts.append(t_boost)
    index = cv_errors.index(min(cv_errors))
    t_opt = t_opts[index]
    return t_opt


def parameter_t_boost_forplot(x_tra, y_tra, f):
    errors, t_opts = [], []
    for i in range(1,31):  # if d=5，range(1, 41), 2 *
        t_boost = int(i*2)
        print('t_boost:', t_boost)
        cv = cv_t_boost(x_tra, y_tra, f, t_boost)
        errors.append(cv)
        t_opts.append(t_boost)
    return errors, t_opts


# in equal-sized setting, N=2000，the number of local agent =2，4，6...,40 ------------------
# partition
def opt_local_t(x_tra, y_tra, m, f, j):
    b = mblocks(x_tra, y_tra, m)
    b_x = np.array(b.loc[j]['block_x'])
    b_y = np.array(b.loc[j]['block_y'])
    t_opt = global_parameter_t_boost(b_x, b_y, f)
    return t_opt


# (2) opt_t for each block of a certain fen_num / division
def local_parameter_t(x_tra, y_tra, num_agent, f):
    t_adapt_blocks = []
    for j in range(num_agent):
        t0 = opt_local_t(x_tra, y_tra, num_agent, f, j)
        print('when num_agent = %s, block = %s, t_adapt = %s' % (num_agent, j, t0))
        t_adapt_blocks.append(t0)
    return t_adapt_blocks


# (3) training parameter t for each blocks
def local_parameter_t_train(x_tra, y_tra, f, fen_num, gap):
    local_ts = {}
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        print('num_agent:', num_agent)
        local_t = local_parameter_t(x_tra, y_tra, num_agent, f)
        local_t = np.array(local_t)
        local_t = np.squeeze(local_t)
        local_ts[i] = local_t
    return local_ts



# in unequal-sized setting, N=2000，the number of local agent =2，4，6...,40 ------------------
# partition
def opt_local_t_diff(x_tra, y_tra, m, f, j):
    dic = mblocks_diff(x_tra, m)
    b_x = x_tra[dic[j]]
    b_y = y_tra[dic[j]]
    print('-----', b_x.shape)
    t_opt = global_parameter_t_boost(b_x, b_y, f)
    return t_opt


def local_parameter_t_diff(x_tra, y_tra, num_agent, f):
    t_adapt_blocks = []
    for j in range(num_agent):
        t0 = opt_local_t_diff(x_tra, y_tra, num_agent, f, j)
        print('when num_agent = %s, block = %s, t_adapt = %s' % (num_agent, j, t0))
        t_adapt_blocks.append(t0)
    return t_adapt_blocks


def local_parameter_t_train_diff(x_tra, y_tra, f, fen_num, gap):
    local_ts = {}
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        print('num_agent:', num_agent)
        local_t = local_parameter_t_diff(x_tra, y_tra, num_agent, f)
        local_t = np.array(local_t)
        local_t = np.squeeze(local_t)
        local_ts[i] = local_t
    return local_ts


def LE_boost(x_tra, y_tra, x_tes1, y_tes, m, t1):
    b = mblocks(x_tra, y_tra, m)
    blocks_error = []
    for i in range(m):
        b_x = np.array(b.loc[i]['block_x'])
        b_y = np.array(b.loc[i]['block_y'])
        x_tes = np.array(x_tes1)
        error = Predictedboost(b_x, b_y, x_tes, y_tes, t1)[1]  #
        blocks_error.append(error)
    return min(blocks_error)


def LE_boost_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, t_para):
    LEs_boost = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        test_error = LE_boost(x_tra, y_tra, x_tes, y_tes, num_agent, t_para)
        LEs_boost.append(test_error)
        print('num_agent: %s, LEs_boost: %s' % (num_agent, test_error))
    return LEs_boost


def AE_boost_opt(x_tra, y_tra, x_tes1, y_tes, m, t1):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        fit = Predictedboost(b_x, b_y, x_tes, y_tes, t1)[0]  #
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)
    fit_global = np.sum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_boost_opt_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, t_para):
    AEs_boost = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        test_error = AE_boost_opt(x_tra, y_tra, x_tes, y_tes, num_agent, t_para)
        AEs_boost.append(test_error)
        print('num_agent: %s, AE_boost: %s' % (num_agent, test_error))
    return AEs_boost


def AE_adapt_boost(x_tra, y_tra, x_tes1, y_tes, m, l_t):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]

        # "adaptive h" or "log h", choose one
        # h_adapt[j]: adaptive parameter "h" of jth block in ith fen_num
        #h = h_adapt[j]
        LOG = math.log(len(x_tra), n)
        l_t_log = l_t[j] ** LOG
        l_t_log = min(int(l_t_log), 80)

        print('l_t[j]', l_t[j])
        print('LOG', LOG)
        print('l_t_log', l_t_log)

        fit = Predictedboost(b_x, b_y, x_tes, y_tes, l_t_log)[0]
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)

    fit_global = np.nansum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_adapt_boost_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, l_t_blocks):
    AEs_adapt_boost = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        l_t1 = l_t_blocks[i]
        test_error = AE_adapt_boost(x_tra, y_tra, x_tes, y_tes, num_agent, l_t1)
        AEs_adapt_boost.append(test_error)
        print('when num_agent = %s, test error = %s' % (num_agent, test_error))
        print('-----------------------------------------------------------------------------')
    return AEs_adapt_boost


# AE_active
def AE_active_boost(x_tra, y_tra, x_tes1, y_tes, m, l_t):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    min_active, local_errors = [], []
    active_m = np.zeros(n1)
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]

        LOG = math.log(len(x_tra), n)
        l_t_log = l_t[j] ** LOG
        l_t_log = min(int(l_t_log), 80)

        fit = Predictedboost(b_x, b_y, x_tes, y_tes, l_t_log)[0]  #
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)

        # adopt active rule to filter out invaluable local estimates
        # active = 1 means this block is activated
        active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
        active = np.squeeze(active)
        active_m += active

        fit = fit * active.reshape(n1, 1)
        fit_blocks[:, j] = np.squeeze(fit)
        y_tes1 = np.reshape(y_tes, (-1, 1))               # (1000, ) --> (1000, 1)
        local_error = (np.sum((fit - y_tes1) ** 2)) / n1  # for adapt local estimate, to prove AVM is necessary
        local_errors.append(local_error)

    min_active.append(min(active_m))
    a1 = np.sum(fit_blocks, axis=1)
    b1 = active_m
    fit_global = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)

    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    test_inactive_index = (np.where(active_m == 0))[0]
    # print('index of test samples without active block:', test_inactive_index)
    return average_error, min(local_errors)


def AE_active_boost_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, l_t_blocks):
    AEs_active_boost, Local_errors = [], []
    for i in range(fen_num):
        num_agent = (1 + i) * gap
        l_t1 = l_t_blocks[i]
        a = AE_active_boost(x_tra, y_tra, x_tes, y_tes, num_agent, l_t1)
        AEs_active_boost.append(a[0])
        Local_errors.append(a[1])
        print('when num_agent = %s, AE_log_active = %s' % (num_agent, a[0]))
        print('-----------------------------------------------------------------------------')
    return AEs_active_boost, Local_errors


def LE_boost_adapt(x_tra, y_tra, x_tes1, y_tes, m, l_t):
    dic = mblocks_diff(x_tra, m)
    blocks_error = []
    for j in range(m):
        b_x = x_tra[dic[j]]
        b_y = y_tra[dic[j]]
        x_tes = np.array(x_tes1)
        l_t_1 = l_t[j]
        error = Predictedboost(b_x, b_y, x_tes, y_tes, l_t_1)[1]
        blocks_error.append(error)
    return min(blocks_error)


def LE_boost_adapt_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, l_t_blocks):
    LEs_boost = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        l_t1 = l_t_blocks[i]
        test_error = LE_boost_adapt(x_tra, y_tra, x_tes, y_tes, num_agent, l_t1)
        LEs_boost.append(test_error)
        print('m: %s, LE_adapt: %s' % (num_agent, test_error))
    return LEs_boost


def AE_active_boost_diff(x_tra, y_tra, x_tes1, y_tes, m, l_t):
    dic = mblocks_diff(x_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    min_active, local_errors = [], []
    active_m = np.zeros(n1)
    for j in range(m):
        b_x = x_tra[dic[j]]
        b_y = y_tra[dic[j]]
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]

        LOG = math.log(len(x_tra), n)
        l_t_log = l_t[j] ** LOG
        l_t_log = min(int(l_t_log), 80)

        fit = Predictedboost(b_x, b_y, x_tes, y_tes, l_t_log)[0]
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)

        # adopt active rule to filter out invaluable local estimates
        # active = 1 means this block is activated
        active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
        active = np.squeeze(active)
        active_m += active

        fit = fit * active.reshape(n1, 1)
        fit_blocks[:, j] = np.squeeze(fit)
        y_tes1 = np.reshape(y_tes, (-1, 1))               # (1000, ) --> (1000, 1)
        local_error = (np.sum((fit - y_tes1) ** 2)) / n1  # for adapt local estimate, to prove AVM is necessary
        local_errors.append(local_error)

    min_active.append(min(active_m))
    a1 = np.sum(fit_blocks, axis=1)
    b1 = active_m
    fit_global = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)

    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    test_inactive_index = (np.where(active_m == 0))[0]
    # print('index of test samples without active block:', test_inactive_index)
    return average_error, min(local_errors)


def AE_active_boost_diff_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, l_t_blocks):
    AEs_active_boost, Local_errors = [], []
    for i in range(fen_num):
        num_agent = (1 + i) * gap
        l_t1 = l_t_blocks[i]
        a = AE_active_boost_diff(x_tra, y_tra, x_tes, y_tes, num_agent, l_t1)
        AEs_active_boost.append(a[0])
        Local_errors.append(a[1])
        print('when num_agent = %s, AE_log_active = %s' % (num_agent, a[0]))
        print('-----------------------------------------------------------------------------')
    return AEs_active_boost, Local_errors


'''
5.3 Rf ---------------------------------------------------  
'''
from sklearn.ensemble import RandomForestRegressor

# l_t = 2
# LOG = math.log(2000, 500) # 10.965784284662087
# l_t_log = math.ceil(l_t ** LOG)

def Predictedrf(x_tra, y_tra, x_tes, y_tes, regular_para):
    n, t = len(x_tra), len(x_tes)
    regr_1 = RandomForestRegressor(n_estimators=regular_para, max_depth=5, random_state=0)
    regr_1.fit(x_tra, y_tra)
    y_fit = regr_1.predict(x_tes)
    y_fit = np.squeeze(y_fit)
    average_error = np.sum((y_fit - y_tes) ** 2) / t
    return y_fit, average_error


def cv_t_rf(x, y, f, t_rf):
    kf = KFold(n_splits=f, shuffle=True)
    # kf = KFold(n_splits=f)
    error = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        result = Predictedrf(x_train, y_train, x_test, y_test, t_rf)
        error.append(result[1])
    return sum(error)/f


def global_parameter_t_rf(x_tra, y_tra, f):
    cv_errors, t_opts = [], []
    for i in range(1,41):  # if d=5，range(1, 51), 2 *
        t_rf = int(i*2)
        # print('t_rf:', t_rf)
        cv_error = cv_t_rf(x_tra, y_tra, f, t_rf)
        cv_errors.append(cv_error)
        t_opts.append(t_rf)
    index = cv_errors.index(min(cv_errors))
    t_opt = t_opts[index]
    return t_opt


def parameter_t_rf_forplot(x_tra, y_tra, f):
    errors, t_opts = [], []
    for i in range(1,41):  # if d=5，range(1, 51), 2 *
        t_rf = int(i*2)
        print('t_rf:', t_rf)
        cv = cv_t_rf(x_tra, y_tra, f, t_rf)
        errors.append(cv)
        t_opts.append(t_rf)
    return errors, t_opts



# in equal-sized setting, N=2000，the number of local agent =2，4，6...,40 ------------------
# partition
def opt_local_t_rf(x_tra, y_tra, m, f, j):
    b = mblocks(x_tra, y_tra, m)
    b_x = np.array(b.loc[j]['block_x'])
    b_y = np.array(b.loc[j]['block_y'])
    t_opt = global_parameter_t_rf(b_x, b_y, f)
    return t_opt


# opt_t for each block of a certain fen_num / division
def local_parameter_t_rf(x_tra, y_tra, num_agent, f):
    t_adapt_blocks = []
    for j in range(num_agent):
        t0 = opt_local_t_rf(x_tra, y_tra, num_agent, f, j)
        print('when num_agent = %s, block = %s, t_adapt = %s' % (num_agent, j, t0))
        t_adapt_blocks.append(t0)
    return t_adapt_blocks


# (3) training parameter t for each blocks
def local_parameter_t_train_rf(x_tra, y_tra, f, fen_num, gap):
    local_ts = {}
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        print('num_agent:', num_agent)
        local_t = local_parameter_t_rf(x_tra, y_tra, num_agent, f)
        local_t = np.array(local_t)
        local_t = np.squeeze(local_t)
        local_ts[i] = local_t
    return local_ts



# in unequal-sized setting, N=2000，the number of local agent =2，4，6...,40 ------------------
# partition
def opt_local_t_diff_rf(x_tra, y_tra, m, f, j):
    dic = mblocks_diff(x_tra, m)
    b_x = x_tra[dic[j]]
    b_y = y_tra[dic[j]]
    print('-----', b_x.shape)
    t_opt = global_parameter_t_rf(b_x, b_y, f)
    return t_opt


def local_parameter_t_diff_rf(x_tra, y_tra, num_agent, f):
    t_adapt_blocks = []
    for j in range(num_agent):
        t0 = opt_local_t_diff_rf(x_tra, y_tra, num_agent, f, j)
        print('when num_agent = %s, block = %s, t_adapt = %s' % (num_agent, j, t0))
        t_adapt_blocks.append(t0)
    return t_adapt_blocks


def local_parameter_t_train_diff_rf(x_tra, y_tra, f, fen_num, gap):
    local_ts = {}
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        print('num_agent:', num_agent)
        local_t = local_parameter_t_diff_rf(x_tra, y_tra, num_agent, f)
        local_t = np.array(local_t)
        local_t = np.squeeze(local_t)
        local_ts[i] = local_t
    return local_ts


def LE_rf(x_tra, y_tra, x_tes1, y_tes, m, t1):
    b = mblocks(x_tra, y_tra, m)
    blocks_error = []
    for i in range(m):
        b_x = np.array(b.loc[i]['block_x'])
        b_y = np.array(b.loc[i]['block_y'])
        x_tes = np.array(x_tes1)
        error = Predictedrf(b_x, b_y, x_tes, y_tes, t1)[1]
        blocks_error.append(error)
    return min(blocks_error)


def LE_rf_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, t_para):
    LEs_rf = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        test_error = LE_rf(x_tra, y_tra, x_tes, y_tes, num_agent, t_para)
        LEs_rf.append(test_error)
        print('num_agent: %s, LEs_rf: %s' % (num_agent, test_error))
    return LEs_rf


def AE_rf_opt(x_tra, y_tra, x_tes1, y_tes, m, t1):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        fit = Predictedrf(b_x, b_y, x_tes, y_tes, t1)[0]  #
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)
    fit_global = np.sum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_rf_opt_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, t_para):
    AEs_rf = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        test_error = AE_rf_opt(x_tra, y_tra, x_tes, y_tes, num_agent, t_para)
        AEs_rf.append(test_error)
        print('num_agent: %s, AE_rf: %s' % (num_agent, test_error))
    return AEs_rf


def AE_adapt_rf(x_tra, y_tra, x_tes1, y_tes, m, l_t):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]

        # "adaptive h" or "log h", choose one
        # h_adapt[j]: adaptive parameter "h" of jth block in ith fen_num
        #h = h_adapt[j]
        LOG = math.log(len(x_tra), n)
        l_t_log = l_t[j] ** LOG
        l_t_log = min(int(l_t_log), 100)

        print('l_t[j]', l_t[j])
        print('LOG', LOG)
        print('l_t_log', l_t_log)

        fit = Predictedrf(b_x, b_y, x_tes, y_tes, l_t_log)[0]
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)
        fit_blocks[:, j] = np.squeeze(fit)

    fit_global = np.nansum(fit_blocks, axis=1) / m
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def AE_adapt_rf_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, l_t_blocks):
    AEs_adapt_rf = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        l_t1 = l_t_blocks[i]
        test_error = AE_adapt_rf(x_tra, y_tra, x_tes, y_tes, num_agent, l_t1)
        AEs_adapt_rf.append(test_error)
        print('when num_agent = %s, test error = %s' % (num_agent, test_error))
        print('-----------------------------------------------------------------------------')
    return AEs_adapt_rf


# AE_active
def AE_active_rf(x_tra, y_tra, x_tes1, y_tes, m, l_t):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    min_active, local_errors = [], []
    active_m = np.zeros(n1)
    for j in range(m):
        b_x = np.array(b.loc[j]['block_x'])
        b_y = np.array(b.loc[j]['block_y'])
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]

        LOG = math.log(len(x_tra), n)
        l_t_log = l_t[j] ** LOG
        l_t_log = min(int(l_t_log), 100)

        fit = Predictedrf(b_x, b_y, x_tes, y_tes, l_t_log)[0]  #
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)

        # adopt active rule to filter out invaluable local estimates
        # active = 1 means this block is activated
        active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
        active = np.squeeze(active)
        active_m += active

        fit = fit * active.reshape(n1, 1)
        fit_blocks[:, j] = np.squeeze(fit)
        y_tes1 = np.reshape(y_tes, (-1, 1))               # (1000, ) --> (1000, 1)
        local_error = (np.sum((fit - y_tes1) ** 2)) / n1  # for adapt local estimate, to prove AVM is necessary
        local_errors.append(local_error)

    min_active.append(min(active_m))
    a1 = np.sum(fit_blocks, axis=1)
    b1 = active_m
    fit_global = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)

    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    test_inactive_index = (np.where(active_m == 0))[0]
    # print('index of test samples without active block:', test_inactive_index)
    return average_error, min(local_errors)


def AE_active_rf_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, l_t_blocks):
    AEs_active_rf, Local_errors = [], []
    for i in range(fen_num):
        num_agent = (1 + i) * gap
        l_t1 = l_t_blocks[i]
        a = AE_active_rf(x_tra, y_tra, x_tes, y_tes, num_agent, l_t1)
        AEs_active_rf.append(a[0])
        Local_errors.append(a[1])
        print('when num_agent = %s, AE_log_active = %s' % (num_agent, a[0]))
        print('-----------------------------------------------------------------------------')
    return AEs_active_rf, Local_errors


def LE_rf_adapt(x_tra, y_tra, x_tes1, y_tes, m, l_t):
    dic = mblocks_diff(x_tra, m)
    blocks_error = []
    for j in range(m):
        b_x = x_tra[dic[j]]
        b_y = y_tra[dic[j]]
        x_tes = np.array(x_tes1)    # ensure x_tes.shape: (t, d) each time
        l_t_1 = l_t[j]
        error = Predictedrf(b_x, b_y, x_tes, y_tes, l_t_1)[1]  #
        blocks_error.append(error)
    return min(blocks_error)


def LE_rf_adapt_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, l_t_blocks):
    LEs_rf = []
    for i in range(fen_num):
        num_agent = (i + 1) * gap
        l_t1 = l_t_blocks[i]
        test_error = LE_rf_adapt(x_tra, y_tra, x_tes, y_tes, num_agent, l_t1)
        LEs_rf.append(test_error)
        print('m: %s, LE_adapt: %s' % (num_agent, test_error))
    return LEs_rf


def AE_active_rf_diff(x_tra, y_tra, x_tes1, y_tes, m, l_t):
    dic = mblocks_diff(x_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    min_active, local_errors = [], []
    active_m = np.zeros(n1)
    for j in range(m):
        b_x = x_tra[dic[j]]
        b_y = y_tra[dic[j]]
        x_tes = np.array(x_tes1)
        n = b_x.shape[0]

        LOG = math.log(len(x_tra), n)
        l_t_log = l_t[j] ** LOG
        l_t_log = min(int(l_t_log), 100)

        fit = Predictedrf(b_x, b_y, x_tes, y_tes, l_t_log)[0]
        fit = np.reshape(fit, (-1, 1))
        fit = np.nan_to_num(fit)

        # adopt active rule to filter out invaluable local estimates
        # active = 1 means this block is activated
        active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
        active = np.squeeze(active)
        active_m += active

        fit = fit * active.reshape(n1, 1)
        fit_blocks[:, j] = np.squeeze(fit)
        y_tes1 = np.reshape(y_tes, (-1, 1))               # (1000, ) --> (1000, 1)
        local_error = (np.sum((fit - y_tes1) ** 2)) / n1  # for adapt local estimate, to prove AVM is necessary
        local_errors.append(local_error)

    min_active.append(min(active_m))
    a1 = np.sum(fit_blocks, axis=1)
    b1 = active_m
    fit_global = np.divide(a1, b1, out=np.zeros_like(a1, dtype=np.float64), where=b1 != 0)

    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    test_inactive_index = (np.where(active_m == 0))[0]
    # print('index of test samples without active block:', test_inactive_index)
    return average_error, min(local_errors)


def AE_active_rf_diff_test(x_tra, y_tra, x_tes, y_tes, fen_num, gap, l_t_blocks):
    AEs_active_rf, Local_errors = [], []
    for i in range(fen_num):
        num_agent = (1 + i) * gap
        l_t1 = l_t_blocks[i]
        a = AE_active_rf_diff(x_tra, y_tra, x_tes, y_tes, num_agent, l_t1)
        AEs_active_rf.append(a[0])
        Local_errors.append(a[1])
        print('when num_agent = %s, AE_log_active = %s' % (num_agent, a[0]))
        print('-----------------------------------------------------------------------------')
    return AEs_active_rf, Local_errors


'''
5.4 Hybrid algorithm of Krr, AdaBoost and Rf ------------------------------------------------  
'''
def hybrid_algorithm(x_tra, y_tra, x_tes1, y_tes, m, fen_now, adapt_krr, adapt_boost, adapt_rf, d):
    b = mblocks(x_tra, y_tra, m)
    n1 = len(x_tes1)
    fit_blocks = np.zeros((n1, m))
    active_m = np.zeros(n1)
    random.seed(0)
    for i in range(m):
        assign = random.randint(1, 3)
        b_x = np.array(b.loc[i]['block_x'])
        b_y = np.array(b.loc[i]['block_y'])
        x_tes = np.array(x_tes1)              # ensure x_tes.shape: (t, d) each time
        n = b_x.shape[0]
        t = x_tes.shape[0]
        LOG = math.log(len(x_tra), n)         # logarithmic operator

        # krr
        if assign == 1:
            #print('epan')
            lambda1 = adapt_krr[fen_now][i] ** LOG
            fit = Predictedkrr(b_x, b_y, x_tes, y_tes, d, lambda1)[0]  # KRR
            fit = np.reshape(fit, (-1, 1))
            fit = np.nan_to_num(fit)

            # qualifing
            active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
            active = np.squeeze(active)
            active_m += active
            fit = fit * active.reshape(n1, 1)
            fit_blocks[:, i] = np.squeeze(fit)

        # boost, min(int(l_t_log), 80)
        if assign == 2:
            t1 = min(int(adapt_boost[fen_now][i] ** LOG), 80)
            fit = Predictedboost(b_x, b_y, x_tes, y_tes, t1)[0]
            fit = np.reshape(fit, (-1, 1))
            fit = np.nan_to_num(fit)

            # qualifing
            active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
            active = np.squeeze(active)
            active_m += active
            fit = fit * active.reshape(n1, 1)
            fit_blocks[:, i] = np.squeeze(fit)

        # rf,min(int(l_t_log), 100)
        if assign == 3:
            t2 = min(int(adapt_rf[fen_now][i] ** LOG), 100)
            fit = Predictedrf(b_x, b_y, x_tes, y_tes, t2)[0]
            fit = np.reshape(fit, (-1, 1))
            fit = np.nan_to_num(fit)

            # qualifing
            active = np.where(abs(fit) >= 1 / len(x_tra), 1, 0)
            active = np.squeeze(active)
            active_m += active
            fit = fit * (active.reshape(n1, 1))
            fit_blocks[:, i] = np.squeeze(fit)
            # fit_blocks = np.nan_to_num(fit_blocks)

    # print('active m after knn', active_m[:5])
    a3 = np.sum(fit_blocks, axis=1)
    b3 = active_m
    fit_global = np.divide(a3, b3, out=np.zeros_like(a3, dtype=np.float64), where=b3 != 0)
    average_error = np.sum((fit_global - y_tes) ** 2) / n1
    return average_error


def hybrid_algorithm_test(x_tra, y_tra, x_tes, y_tes, adapt_krr, adapt_boost, adapt_rf, d, gap, fen_num):
    Hybrid_Error= []
    for i in range(fen_num):
        m = (i + 1) * gap
        fen_now = i
        print('fen_now: %s, m:%s' % (fen_now, m))
        hybrid_error = hybrid_algorithm(x_tra, y_tra, x_tes, y_tes, m, fen_now, adapt_krr, adapt_boost, adapt_rf, d)
        Hybrid_Error.append(hybrid_error)
        print('Hybrid_Error:', Hybrid_Error)
    return Hybrid_Error





















































































































