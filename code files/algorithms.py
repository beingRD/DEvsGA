# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev
# All rights reserved.
#
# This algorithms.py file is part of the Advanced Optimization project (midterm) for the
# university course at Laurentian University.
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


def differential_evolution(objective_function, D, Np=100, Cr=0.9, F=0.8, max_NFC=3000):
    pop = np.random.uniform(-10, 10, (Np, D)) #Population
    fitness_val = np.array([objective_function(individual) for individual in pop])
    bf_error = [] #Best Fitness Error
    nfc = 0

    while nfc < max_NFC:
        for i in range(Np):
            indices = list(range(Np))
            indices.remove(i)
            a, b, c = np.random.choice(indices, size=3, replace=False)
            trial_vector = pop[a] + F * (pop[b] - pop[c])
            crossover_mask = np.random.rand(D) < Cr
            offspring = np.where(crossover_mask, trial_vector, pop[i])

            offspring_fitness = objective_function(offspring)
            nfc += 1
            if offspring_fitness < fitness_val[i]:
                pop[i] = offspring
                fitness_val[i] = offspring_fitness

            if nfc >= max_NFC:
                break

        bf_error.append(np.min(fitness_val))

    return bf_error


# Genetic Algorithm (GA)
def genetic_algorithm(objective_function, D, Np=100, Cr=0.9, pm=0.01, max_NFC=3000):
    pop = np.random.uniform(-10, 10, (Np, D)) #Population
    fitness_val = np.array([objective_function(individual) for individual in pop])
    bf_error = []
    nfc = 0

    while nfc < max_NFC:
        offspring = np.empty_like(pop)
        for i in range(Np):
            indices = list(range(Np))
            indices.remove(i)
            a, b, c = np.random.choice(indices, size=3, replace=False)
            crossover_mask = np.random.rand(D) < Cr
            offspring[i] = np.where(crossover_mask, pop[a] + pm * (pop[b] - pop[c]), pop[i])

        offspring_fitness = np.array([objective_function(individual) for individual in offspring])
        nfc += Np

        for i in range(Np):
            if offspring_fitness[i] < fitness_val[i]:
                pop[i] = offspring[i]
                fitness_val[i] = offspring_fitness[i]

            if nfc >= max_NFC:
                break

        bf_error.append(np.min(fitness_val))

    return bf_error
