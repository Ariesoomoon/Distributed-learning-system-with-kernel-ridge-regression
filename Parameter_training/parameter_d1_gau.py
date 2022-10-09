import os, time
from Function_same_block import generate_data, global_parameter_h, local_parameter_h_test, local_parameter_h
# from Function_diff_block import generate_data, global_parameter_h, local_parameter_h_test
import numpy as np
time_start = time.time()



'''
1. Initialization
'''
f = 5
trails = 20
train, test, d = 10000, 1000, 1
fen_num, gap = 70, 5
p1, p2 = 0.25, 0.75
s = 2
# trails_d1_gau2 = {}
# global_h = []
local_h = {}


loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', allow_pickle=True)
trails_d1_gau2 = loadData.tolist()
print(trails_d1_gau2.keys())


'''2. Calculating training parameter for global and local: 20 random seed for 20 trails'''
for trail in range(trails):
    np.random.seed(trail)
    print('------------------------------------------------ trail:', trail + 1)
    X_train, y_train, X_test, y_test = generate_data(train, test, d)
    print('\nThe first training data:\nX_train:%s,y_train:%s' % (X_train[0:1], y_train[0:1]))
    # gh_trail1 = global_parameter_h(X_train, y_train, f, d)
    # gh_trail1 = int(gh_trail1)
    # global_h.append(gh_trail1)
    # print('global_h:', global_h)

    lh_trail1 = local_parameter_h_test(X_train, y_train, f, d, fen_num, gap)
    local_h[trail] = lh_trail1
    print('local_h:', local_h)


'''
3. save parameters
'''
# trails_d1_gau2['global_h'] = global_h
# trails_d1_gau2['local_h'] = local_h
trails_d1_gau2['local_h_diff'] = local_h
np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', trails_d1_gau2)
print('--------------------------------------------------------------save trails_d1_gau2.npy done')




''' 
   The Cumulative Effect of Clients in the Collaborative Diagnosis System:
   the number of local agents m: 5,10,20; 
   the training size N: 1000,2000,3000,...,10000 
   remember to change the value of 'num_agent' and 'local_hs_N_m5'
'''

'''
4. Initialization
'''
f = 5
trails = 20
p1, p2 = 0.25, 0.75
s = 2
num_agent = 10  # m=5, 10, 20

loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', allow_pickle=True)
trails_d1_gau2 = loadData.tolist()
print(trails_d1_gau2.keys())

global_h = np.empty(shape=(10, 20))
local_h = {}


'''
5. Calculating
'''
for trail in range(trails):
    np.random.seed(trail)
    print('------------------------------------------------ trail:', trail + 1)
    global_h1 = []
    local_h1 = {}

    for i in range(1, 11):
        train, test, d = 1000 * i, 1000, 1  # N=1000,2000,3000,...,10000
        print('------------------------------------------------ N:', 1000 * i)
        X_train, y_train, X_test, y_test = generate_data(train, test, d)
        # h1 = global_parameter_h(X_train, y_train, f, d)
        # global_h1.append(h1)

        lh1 = local_parameter_h(X_train, y_train, num_agent, f, d)
        local_h1[i] = lh1

    # global_h1 = np.array(global_h1)
    # global_h[:, trail] = np.squeeze(global_h1)

    local_h[trail] = local_h1



'''
6. save MSE
'''
# trails_d1_gau2['global_h_Nchange'] = global_h
# trails_d1_gau2['local_h_m5_Nchange'] = local_h
trails_d1_gau2['local_h_m10_Nchange'] = local_h
# trails_d1_gau2['local_h_m20_Nchange'] = local_h

np.save(os.path.dirname(os.getcwd()) + '/Result_data/trails_d1_gau2.npy', trails_d1_gau2)
print('--------------------------------------------------------------save trails_d1_gau2.npy done')
time_total = time.time() - time_start
print('runing time:', time_total)


