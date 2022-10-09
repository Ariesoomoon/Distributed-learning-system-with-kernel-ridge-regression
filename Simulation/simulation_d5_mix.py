import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d5_hybrid.npy', allow_pickle=True)
appendix_d5_hybrid = loadData.tolist()
print(appendix_d5_hybrid.keys())

AE_active_gau_mean = np.mean(loadData.tolist()['AE_active_gau'], axis=1)
AE_active_krr_mean = np.mean(loadData.tolist()['AE_active_krr'], axis=1)
AE_active_boost_mean = np.mean(loadData.tolist()['AE_active_boost'], axis=1)
AE_active_rf_mean = np.mean(loadData.tolist()['AE_active_rf'], axis=1)

LE_opt_gau_mean = np.mean(loadData.tolist()['LE_opt_gau'], axis=1)
LE_opt_krr_mean = np.mean(loadData.tolist()['LE_opt_krr'], axis=1)
LE_opt_boost_mean = np.mean(loadData.tolist()['LE_opt_boost'], axis=1)
LE_opt_rf_mean = np.mean(loadData.tolist()['LE_opt_rf'], axis=1)

AE_active_hybrid_mean = np.mean(loadData.tolist()['AE_active_hybrid'], axis=1)
GEs_boost_mean = [np.mean(loadData.tolist()['GE_boost'])]*20
print(GEs_boost_mean)


'''
figure 1: compare the AE_log_active on each algorithm
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
num_agent = [i for i in range(2, 42, 2)]
ax.plot(num_agent, AE_active_krr_mean, c='forestgreen', linestyle='--', linewidth=2.0)
ax.plot(num_agent, AE_active_boost_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(num_agent, AE_active_rf_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax.plot(num_agent, AE_active_hybrid_mean, c='brown', linestyle='--', linewidth=2.0)
ax.plot(num_agent, GEs_boost_mean, c='black', linestyle='--', linewidth=2.0)
# plt.ylim(0.00001, 0.00023)
plt.legend(['$AE_{active, Krr}$', '$AE_{active, Boost}$', '$AE_{active, Rf}$', '$AE_{active, Hybrid}$', '$GE_{Boost}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents ($d$=5)', fontsize='13')
ax.set_ylabel('MSE', fontsize='13')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/hybrid_d5_AE_compare.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()

# +logscale
left, bottom, width, height = 0.6, 0.7, 0.3, 0.25
ax1 = fig.add_axes([left, bottom, width, height])
# ax1.grid(linestyle='-.')
num_agent = [i for i in range(2, 42, 2)]
ax1.plot(num_agent, AE_active_krr_mean, c='forestgreen', linestyle='--', linewidth=2.0)
ax1.plot(num_agent, AE_active_boost_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax1.plot(num_agent, AE_active_rf_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax1.plot(num_agent, AE_active_hybrid_mean, c='brown', linestyle='--', linewidth=2.0)
ax1.plot(num_agent, GEs_boost_mean, c='black', linestyle='--', linewidth=2.0)
ax1.set_ylabel('log(MSE)')
plt.yscale('log')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/hybrid_d5_AE_compare.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


'''
figure 2: prove the effective of hybrid algorithm. Compare the LE on each algorithm, GE as our baseline.
'''
num_agent = [i for i in range(2, 42, 2)]
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
ax.plot(num_agent, LE_opt_krr_mean, c='forestgreen', linestyle='--', linewidth=2.0)
ax.plot(num_agent, LE_opt_boost_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(num_agent, LE_opt_rf_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax.plot(num_agent, AE_active_hybrid_mean, c='brown', linestyle='--', linewidth=2.0)
# ax.plot(num_agent, GE_gau_mean, c='black', linestyle='--', linewidth=2.0)
plt.legend(['$LE_{opt, Krr}$', '$LE_{opt, Boost}$', '$LE_{opt, Rf}$', '$AE_{active, Hybrid}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents ($d$=5)', fontsize='13')
ax.set_ylabel('MSE', fontsize='13')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/hybrid_d5_LE_compare.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()

# # +logscale
# left, bottom, width, height = 0.6, 0.7, 0.3, 0.25
# ax1 = fig.add_axes([left, bottom, width, height])
# # ax1.grid(linestyle='-.')
# num_agent = [i for i in range(2, 42, 2)]
# ax1.plot(num_agent, LE_opt_krr_mean, c='forestgreen', linestyle='--', linewidth=2.0)
# ax1.plot(num_agent, LE_opt_boost_mean, c='royalblue', linestyle=':', linewidth=2.0)
# ax1.plot(num_agent, LE_opt_rf_mean, c='royalblue', linestyle='--', linewidth=2.0)
# ax1.plot(num_agent, AE_active_hybrid_mean, c='brown', linestyle='--', linewidth=2.0)
# ax1.set_ylabel('log(MSE)')
# plt.yscale('log')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/hybrid_d5_LE_compare.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()




