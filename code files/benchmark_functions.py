# --------------------------------------------------------------------------------
# Copyright (c) 2023, Rishabh Dev
# All rights reserved.
#
# This benchmark_functions.py file is part of the Advanced Optimization project (midterm)
# for the university course at Laurentian University.
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


# Benchmark Functions
def high_conditioned_elliptic(x):
    if isinstance(x, np.ndarray):
        return np.sum(100 * np.square(x))
    else:
        return 100 * np.square(x)


def bent_cigar(x):
    if isinstance(x, np.ndarray):
        return x[0] ** 2 + 1e6 * np.sum(np.square(x[1:]))
    else:
        return x ** 2


def discus(x):
    if isinstance(x, np.ndarray):
        return 1e6 * x[0] ** 2 + np.sum(np.square(x[1:]))
    else:
        return 1e6 * x ** 2


def rosenbrocks(x):
    if isinstance(x, np.ndarray):
        return np.sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)])
    else:
        return 0


def ackleys(x):
    if isinstance(x, np.ndarray):
        num = len(x)
        sum_1 = -0.2 * np.sqrt(np.sum(x ** 2) / num)
        sum_2 = np.sum(np.cos(2 * np.pi * x)) / num
        return -20 * np.exp(sum_1) - np.exp(sum_2) + 20 + np.exp(1)
    else:
        return -20 * np.exp(-0.2 * np.sqrt(x ** 2)) - np.exp(np.cos(2 * np.pi * x)) + 20 + np.exp(1)


def weierstrass(x):
    if isinstance(x, np.ndarray):
        a = 0.5
        b = 3
        it_max = 20
        sum_1 = 0
        for i in range(len(x)):
            sum_2 = 0
            for j in range(it_max + 1):
                sum_2 += a * j * np.cos(2 * np.pi * b * j * (x[i] + 0.5))
            sum_1 += sum_2
        sum_3 = 0
        for j in range(it_max + 1):
            sum_3 += a * j * np.cos(2 * np.pi * b * j * 0.5)
        return sum_1 - len(x) * sum_3
    else:
        a = 0.5
        b = 3
        it_max = 20
        sum_1 = 0
        sum_2 = 0
        for j in range(it_max + 1):
            sum_2 += a * j * np.cos(2 * np.pi * b * j * (x + 0.5))
        sum_1 += sum_2
        sum_3 = 0
        for j in range(it_max + 1):
            sum_3 += a * j * np.cos(2 * np.pi * b * j * 0.5)
        return sum_1 - sum_3


def griewanks(x):
    if isinstance(x, np.ndarray):
        return np.sum(x ** 2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1
    else:
        return x ** 2 / 4000 - np.cos(x / np.sqrt(np.arange(1, 2))) + 1


def rastrigins(x):
    if isinstance(x, np.ndarray):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    else:
        return 10 * len([x]) + x ** 2 - 10 * np.cos(2 * np.pi * x)
