import os, time
from Function_same_block import generate_data, hybrid_algorithm_test
#from Function_diff_block import generate_data, hybrid_algorithm_test, MSE_pri_TQMA, LE_hybrid_algorithm_test
import numpy as np
time_start = time.time()



'''
1.  load parameters of each algorithm, and save all of them in appendix_d1_hybrid.npy
'''
loadData_gau = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', allow_pickle=True)
loadData_krr = np.load(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d1_krr.npy', allow_pickle=True)
loadData_boost = np.load(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d1_boost.npy', allow_pickle=True)
loadData_rf = np.load(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d1_rf.npy', allow_pickle=True)

# appendix_d1_hybrid = {}
# appendix_d1_hybrid['l_h_gau'] = loadData_gau.tolist()['local_h']
# appendix_d1_hybrid['l_lambda_krr'] = loadData_krr.tolist()['l_lambda']
# appendix_d1_hybrid['l_t_boost'] = loadData_boost.tolist()['l_t']
# appendix_d1_hybrid['l_t_rf'] = loadData_rf.tolist()['l_t']
#
# appendix_d1_hybrid['AE_active_gau'] = loadData_gau.tolist()['AE_log_actives']
# appendix_d1_hybrid['AE_active_krr'] = loadData_krr.tolist()['AE_log_actives']
# appendix_d1_hybrid['AE_active_boost'] = loadData_boost.tolist()['AE_log_actives']
# appendix_d1_hybrid['AE_active_rf'] = loadData_rf.tolist()['AE_log_actives']
# appendix_d1_hybrid['GE_gau'] = loadData_gau.tolist()['GEs']
#
# appendix_d1_hybrid['LE_opt_gau'] = loadData_gau.tolist()['LEs']
# appendix_d1_hybrid['LE_opt_krr'] = loadData_krr.tolist()['LEs']
# appendix_d1_hybrid['LE_opt_boost'] = loadData_boost.tolist()['LEs']
# appendix_d1_hybrid['LE_opt_rf'] = loadData_rf.tolist()['LEs']

# loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d1_hybrid.npy', allow_pickle=True)
# appendix_d1_hybrid = loadData.tolist()
# appendix_d1_hybrid['GE_krr'] = loadData_krr.tolist()['GEs']
#
# print(appendix_d1_hybrid.keys())
# np.save(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d1_hybrid.npy', appendix_d1_hybrid)
# print('save appendix_d1_hybrid.npy done')


# GEs_mean of krr     # 0.00013161913271949422  1, we choose this as our baseline in appendix
# GEs_mean of boost   # 0.0004412626741243014   3
# GEs_mean of rf      # 0.0003126548463625126   2



'''
2. Initialization
'''
f = 5
trails = 20
train, test, d = 2000, 1000, 1
fen_num, gap = 20, 2
s = 2
AE_active_hybrid = np.empty(shape=(20, 20))


loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d1_hybrid.npy', allow_pickle=True)
appendix_d1_hybrid = loadData.tolist()
adapt_krr = loadData.tolist()['l_lambda_krr']
adapt_boost = loadData.tolist()['l_t_boost']
adapt_rf = loadData.tolist()['l_t_rf']
print(appendix_d1_hybrid.keys())


'''
3. Calculating the hybrid algorithm
'''
for th in range(trails):
    np.random.seed(th)
    print('                                                                                  trail:', th + 1)
    # (1) create data and load parameter
    X_train, y_train, X_test, y_test = generate_data(train, test, d)
    X_train, X_test = X_train.reshape(-1, 1), X_test.reshape(-1, 1)

    adapt_krr_1 = adapt_krr[th]
    adapt_boost_1 = adapt_boost[th]
    adapt_rf_1 = adapt_rf[th]

    AE = hybrid_algorithm_test(X_train, y_train, X_test, y_test, adapt_krr_1, adapt_boost_1, adapt_rf_1, d, gap, fen_num)
    AE = np.array(AE)
    AE_active_hybrid[:, th] = np.squeeze(AE)


appendix_d1_hybrid['AE_active_hybrid'] = AE_active_hybrid
np.save(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d1_hybrid.npy', appendix_d1_hybrid)
print('save appendix_d1_hybrid.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)


