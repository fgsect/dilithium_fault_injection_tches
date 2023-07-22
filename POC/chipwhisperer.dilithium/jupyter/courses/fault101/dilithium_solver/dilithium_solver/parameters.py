#!/usr/bin/env python3
import numpy as np


class Parameters:
    @staticmethod
    def get_nist_security_level(nist_security_level: int, threads: int = 4):
        if nist_security_level not in [2, 3, 5]:
            raise ValueError(f'NIST Security Level {nist_security_level} does not exist.')
        if nist_security_level == 3:
            return Parameters(q=8380417, n=256, gamma_1=2 ** 19, eta=4, tau=49, k=6, l=5, nist_security_level=nist_security_level, threads=threads)
        elif nist_security_level == 2:
            return Parameters(q=8380417, n=256, gamma_1=2 ** 17, eta=2, tau=39, k=4, l=4, nist_security_level=nist_security_level, threads=threads)
        elif nist_security_level == 5:
            return Parameters(q=8380417, n=256, gamma_1=2 ** 19, eta=2, tau=60, k=8, l=7, nist_security_level=nist_security_level, threads=threads)

    def __init__(self, q: int, n: int, gamma_1: int, eta: int, tau: int, k: int, l: int, nist_security_level: int, threads: int):
        self.q = q
        self.n = n
        self.gamma_1 = gamma_1
        self.eta = eta
        self.tau = tau
        self.k, self.l = k, l

        self._nist_security_level = nist_security_level

        self.__threads = threads

    @property
    def y_range(self):
        return range(-(self.gamma_1 - 1), self.gamma_1 + 1)

    @property
    def s_1_range(self):
        return range(-self.eta, self.eta + 1)

    @property
    def beta(self):
        return self.tau * self.eta

    @property
    def gamma_2(self):
        if self.nist_security_level == 2:
            return (self.q - 1) / 88
        elif self.nist_security_level == 3 or self.nist_security_level == 5:
            return (self.q - 1) / 32
        else:
            raise ValueError()

    @property
    def max_diff(self):
        return 2 * self.beta + self.gamma_1

    @property
    def nist_security_level(self):
        return self._nist_security_level

    @property
    def threads(self):
        return self.__threads

    @property
    def dtype(self):
        return np.int32

