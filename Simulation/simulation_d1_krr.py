import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import os


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/appendix_d1_krr.npy', allow_pickle=True)
appendix_d1_krr = loadData.tolist()
print(appendix_d1_krr.keys())

GEs_mean = [np.mean(appendix_d1_krr['GEs'])] * 20
AEs_mean = np.mean(appendix_d1_krr['AEs'], axis=1)  # average AE, LE, GE of 20 trails, axis=1 means averaging on trail axis
LEs_mean = np.mean(appendix_d1_krr['LEs'], axis=1)
AE_logs_mean = np.mean(appendix_d1_krr['AE_logs'], axis=1)
AE_log_actives_mean = np.mean(appendix_d1_krr['AE_log_actives'], axis=1)
LE_adapt_opts_mean = np.mean(appendix_d1_krr['LE_adapt_opts'], axis=1)
AE_log_actives_diff_mean = np.mean(appendix_d1_krr['AE_log_actives_diff'], axis=1)
print(GEs_mean)


'''
figure 1: h^log is effective
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
num_agent = [i for i in range(2, 42, 2)]
ax.plot(num_agent, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
ax.plot(num_agent, LEs_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(num_agent, AEs_mean, c='royalblue', linestyle='--', linewidth=2.0)
# ax.plot(num_agent, AE_adapts_mean, c='brown', linestyle=':', linewidth=2.0)
ax.plot(num_agent, AE_logs_mean, c='brown', linestyle='--', linewidth=2.0)
# plt.axvline(x=150,ls="-",c="green", linewidth=2.0)
plt.legend(['$GE$', '$LE_{opt}$', '$AE_{opt}$', '$AE_{log}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents ($d$=1)', fontsize='13')
ax.set_ylabel('MSE (Krr)', fontsize='13')
# plt.yscale('log')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/gau2_d1_log.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()


# +logscale
left, bottom, width, height = 0.6, 0.7, 0.3, 0.25
ax1 = fig.add_axes([left, bottom, width, height])
# ax1.grid(linestyle='-.')
num_agent = [i for i in range(2, 42, 2)]
ax1.plot(num_agent, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
ax1.plot(num_agent, LEs_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax1.plot(num_agent, AEs_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax1.plot(num_agent, AE_logs_mean, c='brown', linestyle='--', linewidth=2.0)
ax1.set_ylabel('log(MSE)')
plt.yscale('log')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/krr_d1_paraautonomy.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()



'''
figure 2: active rule is effective
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
num_agent = [i for i in range(2, 42, 2)]
ax.plot(num_agent, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
ax.plot(num_agent, AE_logs_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax.plot(num_agent, AE_log_actives_mean, c='brown', linestyle='--', linewidth=2.0)
ax.set_ylim(0.0001, 0.0003)
plt.legend(['$GE$', '$AE_{log}$', '$AE_{active}$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents ($d$=1)', fontsize='13')
ax.set_ylabel('MSE (Krr)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/krr_d1_activeeffective.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


# +logscale
# left, bottom, width, height = 0.5, 0.7, 0.3, 0.25
# ax1 = fig.add_axes([left, bottom, width, height])
# # ax1.grid(linestyle='-.')
# num_agent = [i for i in range(2, 42, 2)]
# ax1.plot(num_agent, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
# ax1.plot(num_agent, AE_logs_mean, c='royalblue', linestyle=':', linewidth=2.0)
# ax1.plot(num_agent, AE_log_actives_mean, c='brown', linestyle='--', linewidth=2.0)
# ax1.set_ylabel('log(MSE)')
# plt.yscale('log')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/krr_d1_activeeffective.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()




'''
figure 3: MSE of same block size and different block size
'''
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
num_agent = [i for i in range(2, 42, 2)]

ax.plot(num_agent, LE_adapt_opts_mean, c='forestgreen', linestyle='--', linewidth=2.0)
ax.plot(num_agent, AE_log_actives_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax.plot(num_agent, AE_log_actives_diff_mean, c='brown', linestyle='--', linewidth=2.0)
ax.plot(num_agent, GEs_mean, c='black', linestyle='-.', linewidth=2.0)
# plt.ylim(0.00002, 0.00020)
plt.legend(['$LE_{adapt}$', '$AE_{active,same}$', '$AE_{active,diff}$', '$GE$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of local agents ($d$=1)', fontsize='13')
ax.set_ylabel('MSE (Krr)', fontsize='13')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/krr_d1_same_diff_compare.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()


