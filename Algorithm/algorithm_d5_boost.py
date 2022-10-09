import os, time
from Function_same_block import generate_data, GE_gaussiank, LE_gaussiank_test, AE_gaussiank_test, AE_adapt_gaussiank_test, \
    AE_active_gaussiank_test, Predictedboost,LE_boost_test, AE_boost_opt_test, \
    AE_adapt_boost_test,AE_active_boost_test,LE_boost_adapt_test,AE_active_boost_diff_test

# from Function_diff_block import generate_data, GE_gaussiank, LE_gaussiank_test, AE_gaussiank_test, \
#     AE_adapt_gaussiank_test, AE_active_gaussiank_test, LE_gaussiank_adapt_test
import numpy as np
time_start = time.time()


'''
1. Initialization
'''
f = 5                  # 5-fold cross-validation
trails = 20            # 20 trails for averaging
train, test, d = (2000,5), (1000,5), 5  # training size; testing size; dimensional of data
fen_num, gap = 20, 2   # the number of agents: 2*1, 2*2,...,2*20 (that is, 2,4,...,40)


# load the localization parameters
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d5_boost.npy', allow_pickle=True)
appendix_d5_boost = loadData.tolist()
print(appendix_d5_boost.keys())
g_ts = appendix_d5_boost['g_t']
l_ts = appendix_d5_boost['l_t']
l_ts_diff = appendix_d5_boost['l_t_diff']


# 20 type of divisions, 20trails
GEs = []
AEs = np.empty(shape=(20, 20))
LEs = np.empty(shape=(20, 20))
AE_adapts = np.empty(shape=(20, 20))
AE_logs = np.empty(shape=(20, 20))
AE_log_actives = np.empty(shape=(20, 20))
LE_adapt_opts = np.empty(shape=(20, 20))
AE_log_actives_diff = np.empty(shape=(20, 20))


'''
2. Calculating different types of MSE for 20 trails
'''
for th in range(trails):
    np.random.seed(th)
    print('                                                                                  trail:', th + 1)
    # (1) create data and load parameter
    X_train, y_train, X_test, y_test = generate_data(train, test, d)

    g_t = g_ts[th]
    l_t = l_ts[th]
    l_t_diff = l_ts_diff[th]

    # (2) calculating
    GE = Predictedboost(X_train, y_train, X_test, y_test, g_t)[1]
    GEs.append(GE)

    LE = LE_boost_test(X_train, y_train, X_test, y_test, fen_num, gap, g_t)
    LE = np.array(LE)
    LEs[:, th] = np.squeeze(LE)


    AE = AE_boost_opt_test(X_train, y_train, X_test, y_test, fen_num, gap, g_t)
    AE = np.array(AE)
    AEs[:, th] = np.squeeze(AE)


    AE_log = AE_adapt_boost_test(X_train, y_train, X_test, y_test, fen_num, gap, l_t)
    AE_log = np.array(AE_log)
    AE_logs[:, th] = np.squeeze(AE_log)


    AE_log_active = AE_active_boost_test(X_train, y_train, X_test, y_test, fen_num, gap, l_t)[0]
    AE_log_active = np.array(AE_log_active)
    AE_log_actives[:, th] = np.squeeze(AE_log_active)


    # when local agents hold different data sizes
    LE_adapt_opt = LE_boost_adapt_test(X_train, y_train, X_test, y_test, fen_num, gap, l_t_diff)
    LE_adapt_opt = np.array(LE_adapt_opt)
    LE_adapt_opts[:, th] = np.squeeze(LE_adapt_opt)

    AE_log_active_diff = AE_active_boost_diff_test(X_train, y_train, X_test, y_test, fen_num, gap, l_t_diff)[0]
    AE_log_active_diff = np.array(AE_log_active_diff)
    AE_log_actives_diff[:, th] = np.squeeze(AE_log_active_diff)


'''
3. save MSE
'''
appendix_d5_boost['GEs'] = GEs
# appendix_d5_boost['GEs_new'] = GEs
appendix_d5_boost['LEs'] = LEs
appendix_d5_boost['AEs'] = AEs
appendix_d5_boost['AE_logs'] = AE_logs
appendix_d5_boost['AE_log_actives'] = AE_log_actives

appendix_d5_boost['LE_adapt_opts'] = LE_adapt_opts
appendix_d5_boost['AE_log_actives_diff'] = AE_log_actives_diff


np.save(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d5_boost.npy', appendix_d5_boost)
print('--------------------------------------------------------------save appendix_d5_boost.npy done')
print(appendix_d5_boost.keys())
time_total = time.time() - time_start
print('runing time:', time_total)





