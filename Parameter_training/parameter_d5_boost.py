
import os, time
from Function_same_block_linshi_boostd5 import generate_data, global_parameter_h, local_parameter_h_test, parameter_lambda_krr_forplot,\
    global_parameter_lambda_krr, local_parameter_lambda_train, local_parameter_lambda_train_diff, \
    parameter_t_boost_forplot,global_parameter_t_boost, local_parameter_t_train, local_parameter_t_train_diff
#from Function_diff_block import generate_data, global_parameter_h, local_parameter_h_test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
time_start = time.time()


# appendix_d5_boost= {}
# np.save('/Users/liuxiaotong/Documents/machine learning/Adaptive learning system(nowuse_addkrr_boost_rf)/Result_data/appendix_d5_boost.npy', appendix_d1_krr)
# print('save appendix_d5_boost.npy done')


'''
1. Initialization
'''
f = 5
trails = 20
train, test, d = (2000,5), (1000,5), 5
fen_num, gap = 20, 2


loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d5_boost.npy', allow_pickle=True)
appendix_d5_boost = loadData.tolist()
print(appendix_d5_boost.keys())

# np.save(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d5_boost.npy', appendix_d5_boost)
# print('save appendix_d5_boost.npy done')
# print(appendix_d5_boost.keys())


'''
2. test at first: MSE of AdaBoost with the increasing parameter t
'''
for th in range(2,3):
    np.random.seed(th)
    print('                                                                                  trail:', th + 1)
    # 1. create data and load parameter
    abs_diff_t1s = []
    rl_t1s = []
    X_train, y_train, X_test, y_test = generate_data(train, test, d)

    boost1 = parameter_t_boost_forplot(X_train, y_train, f)
    mse_boost, t_boost = boost1[0], boost1[1]

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(linestyle='-.')
    ax.plot(t_boost, mse_boost, c='black', linestyle='-.', linewidth=1.4)
    ax.set_xlabel('Number of estimators (N=500)', fontsize='13')
    ax.set_ylabel('MSE', fontsize='13')
    plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/Boost_mse_t_1trail_N2000d5.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()



'''
3. Calculating optimal parameter t of AdaBoost
'''
g_t = []
l_t = {}
l_t_diff = {}


for trail in range(trails):
    np.random.seed(trail)
    print('------------------------------------------------ trail:', trail + 1)
    X_train, y_train, X_test, y_test = generate_data(train, test, d)
    # X_train, X_test = X_train.reshape(-1, 1), X_test.reshape(-1, 1)

    g_t_t1 = global_parameter_t_boost(X_train, y_train, f)
    g_t.append(g_t_t1)

    # l_t_t1 = local_parameter_t_train(X_train, y_train, f, fen_num, gap)
    # l_t[trail] = l_t_t1
    #
    # l_t_t2 = local_parameter_t_train_diff(X_train, y_train, f, fen_num, gap)
    # l_t_diff[trail] = l_t_t2



'''
4. save parameters
'''
appendix_d5_boost['g_t'] = g_t # g_t: [42, 12, 50, 30, 48, 64, 16, 68, 12, 96, 86, 74, 96, 6, 86, 74, 88, 76, 34, 98]
# appendix_d5_boost['l_t'] = l_t
# appendix_d5_boost['l_t_diff'] = l_t_diff

np.save(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d5_boost.npy', appendix_d5_boost)
print('--------------------------------------------------------------save appendix_d5_boost.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)  # runing time: 2007.737620830536













































