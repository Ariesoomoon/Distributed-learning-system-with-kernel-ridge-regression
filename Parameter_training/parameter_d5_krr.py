import os, time
from Function_same_block import generate_data, global_parameter_h, local_parameter_h_test, parameter_lambda_krr_forplot,\
    global_parameter_lambda_krr, local_parameter_lambda_train, local_parameter_lambda_train_diff
#from Function_diff_block import generate_data, global_parameter_h, local_parameter_h_test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
time_start = time.time()


# appendix_d5_krr= {}
# np.save('/Users/liuxiaotong/Documents/machine learning/Adaptive learning system(nowuse_addkrr_boost_rf)/Result_data/appendix_d5_krr.npy', appendix_d5_krr)
# print('save appendix_d5_krr.npy done')


'''
1. Initialization
'''
f = 5
trails = 20
train, test, d = (2000,5), (1000,5), 5
fen_num, gap = 20, 2


loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d5_krr.npy', allow_pickle=True)
appendix_d5_krr = loadData.tolist()
print(appendix_d5_krr.keys())

# np.save(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d5_krr.npy', appendix_d5_krr)
# print('save appendix_d5_krr.npy done')
# print(appendix_d5_krr.keys())


'''
2. test at first: MSE of Krr with the increasing parameter lambda
'''
for th in range(1):
    np.random.seed(th)
    print('                                                                                  trail:', th + 1)
    # 1. create data and load parameter
    abs_diff_t1s = []
    rl_t1s = []
    X_train, y_train, X_test, y_test = generate_data(train, test, d)

    krr = parameter_lambda_krr_forplot(X_train, y_train, f, d)
    mse_krr, lambda_krr = krr[0], krr[1]

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(linestyle='-.')
    ax.plot(lambda_krr, mse_krr, c='black', linestyle='-.', linewidth=1.4)
    ax.set_xlabel('$\lambda$ (N=100)', fontsize='13')
    ax.set_ylabel('MSE', fontsize='13')
    plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/KRR_mse_lambda_1trail_N100d5.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()


'''
3. Calculating optimal parameter lambda of Krr
'''
g_lambda = []
l_lambda = {}
l_lambda_diff = {}

for trail in range(trails):
    np.random.seed(trail)
    print('------------------------------------------------ trail:', trail + 1)
    X_train, y_train, X_test, y_test = generate_data(train, test, d)

    g_lam_t1 = global_parameter_lambda_krr(X_train, y_train, f, d)
    g_lambda.append(g_lam_t1)
    print('g_lambda:', g_lambda)

    l_lam_t1 = local_parameter_lambda_train(X_train, y_train, f, d, fen_num, gap)
    l_lambda[trail] = l_lam_t1
    # print('l_lambda:', l_lambda)

    l_lam_t2 = local_parameter_lambda_train_diff(X_train, y_train, f, d, fen_num, gap)
    l_lambda_diff[trail] = l_lam_t2
    # print('l_lambda_diff:', l_lambda_diff)


'''
4. save parameters
'''
appendix_d5_krr['g_lambda'] = g_lambda
appendix_d5_krr['l_lambda'] = l_lambda
appendix_d5_krr['l_lambda_diff'] = l_lambda_diff

np.save(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d5_krr.npy', appendix_d5_krr)
print('--------------------------------------------------------------save appendix_d5_krr.npy done')
print(appendix_d5_krr.keys())
time_total = time.time() - time_start
print('runing time:', time_total)  # 10:03 runing time:


