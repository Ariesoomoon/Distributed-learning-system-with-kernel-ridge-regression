import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import pdb, time, os
import sys


'''
1. load data
'''
loadData = np.load(os.path.dirname(os.getcwd()) + '/Result_data/trails_d5_gau2.npy', allow_pickle=True)
trails_d5_gau2 = loadData.tolist()
print(trails_d5_gau2.keys())


'''
figure: The Cumulative Effect of Clients in the Collaborative Diagnosis System
'''
GEs_Nchange_mean = np.mean(trails_d5_gau2['GEs_Nchange'], axis=1)
AE_active_m5_Nchange_mean = np.mean(trails_d5_gau2['AE_active_m5_Nchange'], axis=1)
AE_active_m10_Nchange_mean = np.mean(trails_d5_gau2['AE_active_m10_Nchange'], axis=1)
AE_active_m20_Nchange_mean = np.mean(trails_d5_gau2['AE_active_m20_Nchange'], axis=1)

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.grid(linestyle='-.')
num_patient = [i for i in range(1000, 10001, 1000)]  # [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

ax.plot(num_patient, GEs_Nchange_mean, c='black', linestyle='-.', linewidth=2.0)
ax.plot(num_patient, AE_active_m5_Nchange_mean, c='brown', linestyle='-.', linewidth=2.0)
ax.plot(num_patient, AE_active_m10_Nchange_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax.plot(num_patient, AE_active_m20_Nchange_mean, c='royalblue', linestyle=':', linewidth=2.0)
plt.legend(['$GE$', '$AE_{active}(m=5)$', '$AE_{active}(m=10)$', '$AE_{active}(m=20)$'], loc='upper left', fontsize='medium')
ax.set_xlabel('Number of clients ($d$=5)', fontsize='13')
ax.set_ylabel('MSE (Gaussian)', fontsize='13')
plt.yscale('log')
# plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/Increasing_clients_d5.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
# plt.show()


# +logscale
left, bottom, width, height = 0.6, 0.7, 0.3, 0.25
ax1 = fig.add_axes([left, bottom, width, height])
# ax1.grid(linestyle='-.')
num_patient = [i for i in range(1000, 10001, 1000)]
ax1.plot(num_patient, GEs_Nchange_mean, c='black', linestyle='-.', linewidth=2.0)
ax1.plot(num_patient, AE_active_m5_Nchange_mean, c='brown', linestyle='-.', linewidth=2.0)
ax1.plot(num_patient, AE_active_m10_Nchange_mean, c='royalblue', linestyle='--', linewidth=2.0)
ax1.plot(num_patient, AE_active_m20_Nchange_mean, c='royalblue', linestyle=':', linewidth=2.0)
ax1.set_ylabel('log(MSE)')
plt.yscale('log')
plt.savefig(os.path.dirname(os.getcwd()) + '/Result_figure/Increasing_clients_d5.pdf', dpi=600, format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()







