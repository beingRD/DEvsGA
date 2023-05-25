# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev
# All rights reserved.
#
# This main_optimization.py file is part of the Advanced Optimization project (midterm) for
# the university course at Laurentian University.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# --------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from benchmark_functions import high_conditioned_elliptic, bent_cigar, discus,rosenbrocks, ackleys, weierstrass, griewanks, rastrigins
from algorithms import genetic_algorithm, differential_evolution

# Control Parameter Settings
Np = 100
Cr = 0.9
F = 0.8
pm = 0.01
all_dimensions = [2, 10, 20]
number_of_generations = 32


# Termination condition: Max_NFC= 3000*D
def max_NFC(D):
    return 3000 * D


# Differential Evolution Algorithm

# Run the experiments
benchmark_functions = [
    high_conditioned_elliptic,
    bent_cigar,
    discus,
    rosenbrocks,
    ackleys,
    weierstrass,
    griewanks,
    rastrigins
]

results = []
errors_data = pd.DataFrame(columns=['Benchmark',
                                    'D',
                                    'Algorithm',
                                    'Mean',
                                    'Best',
                                    'Std'])


for function in benchmark_functions:
    diff_errors = []
    gene_errors = []
    for D in all_dimensions:
        diff_errors_D = []
        gene_errors_D = []
        for run in range(number_of_generations):
            diff_error = differential_evolution(function, D, Np, Cr, F, max_NFC(D))
            gene_error = genetic_algorithm(function, D, Np, Cr, pm, max_NFC(D))
            diff_errors_D.append(diff_error)
            gene_errors_D.append(gene_error)

        diff_errors.append(diff_errors_D)
        gene_errors.append(gene_errors_D)

    # Performance Plots
    for i, D in enumerate(all_dimensions):
        diff_errors_D = np.array(diff_errors[i])
        gene_errors_D = np.array(gene_errors[i])

        diff_mean_error = np.mean(diff_errors_D, axis=0)
        gene_mean_error = np.mean(gene_errors_D, axis=0)

        result = {
            'Function': function.__name__,
            'Dimension': D,
            'DE_Best_Error': np.min(diff_errors_D, axis=1),
            'DE_Mean_Error': diff_mean_error,
            'GA_Best_Error': np.min(gene_errors_D, axis=1),
            'GA_Mean_Error': gene_mean_error
        }

        results.append(result)

        nfc = np.arange(len(diff_mean_error)) * Np

        plt.plot(nfc, diff_mean_error, label='DE', color='purple', linewidth=2)
        plt.plot(nfc, gene_mean_error, label='GA', color='red', linestyle='--', linewidth=2)
        plt.xlabel('NFCs', fontsize=12)
        plt.ylabel('Best Fitness Error So Far', fontsize=12)
        plt.title(f'{function.__name__} , Dimension => {D}', fontsize=14)
        plt.legend()
        plt.show()


# Fill the DataFrame with results
for result in results:
    benchmark = result['Function']
    D = result['Dimension']
    diff_best_error = np.min(result['DE_Best_Error'])
    diff_mean_error = np.mean(result['DE_Mean_Error'])
    diff_std_error = np.std(result['DE_Mean_Error'])
    gene_best_error = np.min(result['GA_Best_Error'])
    gene_mean_error = np.mean(result['GA_Mean_Error'])
    gene_std_error = np.std(result['GA_Mean_Error'])

    errors_data = pd.concat([
        errors_data,
        pd.DataFrame({
            'Benchmark': benchmark,
            'D': D,
            'Algorithm': 'DE',
            'Mean Err': diff_mean_error,
            'Best Err': diff_best_error,
            'Std Err': diff_std_error
        }, index=[0]),
        pd.DataFrame({
            'Benchmark': benchmark,
            'D': D,
            'Algorithm': 'GA',
            'Mean Err': gene_mean_error,
            'Best Err': gene_best_error,
            'Std Err': gene_std_error
        }, index=[0])
    ], ignore_index=True)

# Create DataFrames for each dimension
dfs = []
for dimension in all_dimensions:
    df_dimension = errors_data[errors_data['D'] == dimension]
    dfs.append(df_dimension)

# Display the DataFrames for each dimension
for i, dimension in enumerate(all_dimensions):
    print(f"Errors Data, for Dimension {dimension} =>")
    print(dfs[i])
    print()
    
# Save the DataFrames for each dimension to CSV files
for i, dimension in enumerate(all_dimensions):
    filename = f"dimension_{dimension}_errors.csv"
    dfs[i].to_csv(filename, index=False)
    print(f"Saved DataFrame for Dimension {dimension} to {filename}")
