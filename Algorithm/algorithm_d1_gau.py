import os, time
from Function_same_block import generate_data, GE_gaussiank, LE_gaussiank_test, AE_gaussiank_test, \
    AE_adapt_gaussiank_test, AE_active_gaussiank_test, AE_active_gaussiank
# from Function_diff_block import generate_data, GE_gaussiank, LE_gaussiank_test, AE_gaussiank_test, AE_adapt_gaussiank_test, AE_active_gaussiank_test, LE_gaussiank_adapt_test
import numpy as np
time_start = time.time()

'''
0. remember to choose the Function_same_block.py or Function_diff_block.py, 
and use the corresponding localization parameters: local_h or local_diff_h
'''


'''
1. Initialization
'''
f = 5                  # 5-fold cross-validation
trails = 20            # 20 trails for averaging
train, test, d = 10000, 1000, 1  # training size; testing size; dimensional of data
fen_num, gap = 70, 5   # the number of agents: 5*1, 5*2,...,5*70 (that is, 5,10,...,350)
p1, p2 = 0.25, 0.75    # for tarining localization parameter h
s = 2                  # a parameter of gaussian kernel


# 70 type of divisions, 20trails
GEs = []
AEs = np.empty(shape=(70, 20))
LEs = np.empty(shape=(70, 20))
AE_adapts = np.empty(shape=(70, 20))
AE_logs = np.empty(shape=(70, 20))
AE_log_actives = np.empty(shape=(70, 20))
LE_adapts = np.empty(shape=(70, 20))


# load the localization parameters
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', allow_pickle=True)
trails_d1_gau2 = loadData.tolist()
global_hs = trails_d1_gau2['global_h']
local_hs = trails_d1_gau2['local_h']         # in equal-sized setting
# local_hs = trails_d1_gau2['local_h_diff']  # in unequal-sized setting
print(trails_d1_gau2.keys())



'''
2. Calculating different types of MSE for 20 trails by NWK(gaussian)
'''
for th in range(trails):
    np.random.seed(th)
    print('                                                                                  trail:', th + 1)
    # (1) create data and load localization parameter
    X_train, y_train, X_test, y_test = generate_data(train, test, d)
    global_h = global_hs[th]     # for th trail, select the corresponding global_h[th]
    local_h = local_hs[th]
    print('The first training data, X_train[0]: %s, y_train[0]: %s' % (X_train[0], y_train[0]))
    # print('global_h:', global_h)
    # print('local_h[trail][0]:', local_h[0])
    # print('--------------------------------------------------------------------------------')

    # (2) calculating
    GE = GE_gaussiank(X_train, y_train, X_test, y_test, global_h, d, s)
    GEs.append(GE)

    LE = LE_gaussiank_test(X_train, y_train, X_test, y_test, fen_num, gap, global_h, d, s)
    AE = AE_gaussiank_test(X_train, y_train, X_test, y_test, fen_num, gap, global_h, d, s)             # AE with the global optimal h
    AE_adapt = AE_adapt_gaussiank_test(X_train, y_train, X_test, y_test, d, fen_num, gap, s, local_h)  # AE with the local agent's own optimal h
    a = AE_active_gaussiank_test(X_train, y_train, X_test, y_test, d, fen_num, gap, s, local_h)
    AE_log_active, LE_adapt = a[0], a[1]


    # (3) saving
    LE = np.array(LE)
    LEs[:, th] = np.squeeze(LE)

    AE = np.array(AE)
    AEs[:, th] = np.squeeze(AE)

    AE_adapt = np.array(AE_adapt)
    AE_adapts[:, th] = np.squeeze(AE_adapt)

    AE_log_active = np.array(AE_log_active)
    AE_log_actives[:, th] = np.squeeze(AE_log_active)


    # when local agents hold different data sizes
    # (4) calculate AE_log separately, remember to change h_adapt to h^log by logorithmic mechanism
    AE_log = AE_adapt_gaussiank_test(X_train, y_train, X_test, y_test, d, fen_num, gap, s, local_h)  # with h^log
    AE_log = np.array(AE_log)
    AE_logs[:, th] = np.squeeze(AE_log)

    # (5) calculate LE_adapt
    LE_adapt = LE_gaussiank_adapt_test(X_train, y_train, X_test, y_test, fen_num, gap, local_h, d, s)[0]
    LE_adapt = np.array(LE_adapt)
    LE_adapts[:, th] = np.squeeze(LE_adapt)

    # (6) calculate the data size of the last local agent
    block_size = LE_gaussiank_adapt_test(X_train, y_train, X_test, y_test, fen_num, gap, local_h, d, s)[1]
    block_size = np.array(block_size)
    block_sizes[:, th] = np.squeeze(block_size)



'''
3. save MSE
'''
trails_d1_gau2['GEs'] = GEs
trails_d1_gau2['LEs'] = LEs              # each local agent uses the global optimal localization parameter
trails_d1_gau2['AEs'] = AEs
trails_d1_gau2['AE_adapts'] = AE_adapts
trails_d1_gau2['AE_log_actives'] = AE_log_actives
trails_d1_gau2['AE_logs'] = AE_logs

trails_d1_gau2['LE_adapts'] = LE_adapts  # each local agent uses its own optimal localization parameter
trails_d1_gau2['AE_log_actives_diff'] = AE_log_actives
trails_d1_gau2['last_block_size'] = block_size

np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', trails_d1_gau2)
print('--------------------------------------------------------------save trails_d1_gau2.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)





''' 
4. The Cumulative Effect of Clients in the Collaborative Diagnosis System:
   the number of local agents m: 5,10,20; 
   the training size N: 1000,2000,3000,...,10000 
   
   remember to change the value of 'num_agent' and 'local_hs_N_m5'
'''
f = 5
trails = 20
p1, p2 = 0.25, 0.75
s = 2
GEs = np.empty(shape=(10, 20))
AE_log_actives = np.empty(shape=(10, 20))


# load the localization parameters
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', allow_pickle=True)
trails_d1_gau2 = loadData.tolist()
print(trails_d1_gau2.keys())
global_hs_N = trails_d1_gau2['global_h_Nchange']
local_hs_N_m5 = trails_d1_gau2['local_h_m5_Nchange']
local_hs_N_m10 = trails_d1_gau2['local_h_m10_Nchange']
local_hs_N_m20 = trails_d1_gau2['local_h_m20_Nchange']
num_agent = 5 # m=5, 10, 20


# Calculating different types of MSE for 20 trails by NWK(gaussian)
for trail in range(trails):
    np.random.seed(trail)
    print('------------------------------------------------ trail:', trail + 1)
    ges = []
    aes = []
    for i in range(1, 11):
        train, test, d = 1000 * i, 1000, 1  # N=1000,2000,3000,...,10000
        print('------------------------------------------------ N:', 1000 * i)
        X_train, y_train, X_test, y_test = generate_data(train, test, d)
        global_h = global_hs_N[:, trail][i-1]
        h_adapt = local_hs_N_m5[trail][i]
        # h_adapt = local_hs_N_m10[trail][i]
        # h_adapt = local_hs_N_m20[trail][i]

        # ge = GE_gaussiank(X_train, y_train, X_test, y_test, global_h, d, s)
        # ges.append(ge)

        ae = AE_active_gaussiank(X_train, y_train, X_test, y_test, num_agent, h_adapt, d, s)[0]
        aes.append(ae)

    # ges = np.array(ges)
    # GEs[:, trail] = np.squeeze(ges)

    aes = np.array(aes)
    AE_log_actives[:, trail] = np.squeeze(aes)


## save MSE
# trails_d1_gau2['GEs_Nchange'] = GEs
trails_d1_gau2['AE_active_m5_Nchange'] = AE_log_actives
# trails_d1_gau2['AE_active_m10_Nchange'] = AE_log_actives
# trails_d1_gau2['AE_active_m20_Nchange'] = AE_log_actives


np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', trails_d1_gau2)
print('--------------------------------------------------------------save trails_d1_gau2.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)




