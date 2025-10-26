# Create plot(s) for the different friction results
# The data is stored in a list of dictionaries, where each dictionary contains the results for a different initial condition and friction coefficient.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    # Load the data
    tvlmpc_data_mu10 = np.load('results/tvlmpc_results_mu10.npy', allow_pickle=True)
    kmpc_data_mu10 = np.load('results/kmpccomparison_results_mu10.npy', allow_pickle=True)
    kmpc_data_mu10 = np.load('variable_mu/results/varmu_results_mu10.npy', allow_pickle=True)
    lmpc_data_mu10 = np.load('results/lmpc_results_mu10.npy', allow_pickle=True)
    tvlmpcprojected_data_mu10 = np.load('results/tvlmpcprojected_results_mu10.npy', allow_pickle=True)
    # Convert the data to a DataFrame
    results_arr_mu10 = [tvlmpc_data_mu10, kmpc_data_mu10, lmpc_data_mu10, tvlmpcprojected_data_mu10]
    tvlmpc_data_mu1 = np.load('results/tvlmpc_results_mu1.npy', allow_pickle=True)
    kmpc_data_mu1 = np.load('results/kmpccomparison_results_mu1.npy', allow_pickle=True)
    kmpc_data_mu1 = np.load('variable_mu/results/varmu_results_mu1.npy', allow_pickle=True)
    lmpc_data_mu1 = np.load('results/lmpc_results_mu1.npy', allow_pickle=True)
    tvlmpcprojected_data_mu1 = np.load('results/tvlmpcprojected_results_mu1.npy', allow_pickle=True)
    results_arr_mu1 = [tvlmpc_data_mu1, kmpc_data_mu1, lmpc_data_mu1, tvlmpcprojected_data_mu1]
    # Convert the data to a DataFrame

    # Go through each dictionary and add a 'sum_cost' key
    for entry1 in results_arr_mu10:
        for entry in entry1:
            entry['sum_cost'] = np.sum(entry['poserrsq_data'])
            entry['mpc_sum_cost'] = np.sum(entry['MPC_cost_data'])
    for entry1 in results_arr_mu1:
        for entry in entry1:
            entry['sum_cost'] = np.sum(entry['poserrsq_data'])
            entry['mpc_sum_cost'] = np.sum(entry['MPC_cost_data'])

    tvlmpc_df_mu1 = pd.DataFrame(list(tvlmpc_data_mu1))
    kmpc_df_mu1 = pd.DataFrame(list(kmpc_data_mu1))
    lmpc_df_mu1 = pd.DataFrame(list(lmpc_data_mu1))
    tvlmpcprojected_df_mu1 = pd.DataFrame(list(tvlmpcprojected_data_mu1))
    tvlmpc_df_mu10 = pd.DataFrame(list(tvlmpc_data_mu10))
    kmpc_df_mu10 = pd.DataFrame(list(kmpc_data_mu10))
    lmpc_df_mu10 = pd.DataFrame(list(lmpc_data_mu10))
    tvlmpcprojected_df_mu10 = pd.DataFrame(list(tvlmpcprojected_data_mu10))
    tvlmpc_df_mu1['model'] = 'TVLMPC'
    kmpc_df_mu1['model'] = 'KMPC'
    lmpc_df_mu1['model'] = 'LMPC'
    tvlmpc_df_mu10['model'] = 'TVLMPC'
    tvlmpcprojected_df_mu1['model'] = 'TVLMPCProjected'
    kmpc_df_mu10['model'] = 'KMPC'
    lmpc_df_mu10['model'] = 'LMPC'
    tvlmpcprojected_df_mu10['model'] = 'TVLMPCProjected'
    combined_df = pd.concat([tvlmpc_df_mu1, kmpc_df_mu1, lmpc_df_mu1, tvlmpcprojected_df_mu1, tvlmpc_df_mu10, kmpc_df_mu10, lmpc_df_mu10, tvlmpcprojected_df_mu10], ignore_index=True)
    combined_df = pd.concat([tvlmpc_df_mu1, kmpc_df_mu1, lmpc_df_mu1, tvlmpc_df_mu10, kmpc_df_mu10, lmpc_df_mu10], ignore_index=True)

    # Create a boxplot for the cost
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model', y='sum_cost', hue='mu_t',data=combined_df)
    plt.title('Cost Comparison: Linear Controllers')
    plt.savefig('results/linearizationmodels_cost_comparison.png', dpi=300)
    plt.show()
