#!/usr/bin/env python3
import numpy as np
from math import ceil


class Parameters:
    def __init__(self, nist_security_level: int = 3, threads=4):
        # we use 4 threads as default because this is the amount of virtual processors available on my MacBook
        # https://www.gurobi.com/documentation/9.5/refman/threads.html
        self.__threads = threads
        if nist_security_level == 3:
            self.__nist_security_level = 3
            self.__n = 2048
            self.__B = 2 ** 21 - 1
            self.__h = 40
            # tau is chosen to be 6 as this is the default in sagemath
            # I do not think the BLISS authors mentioned an explicit value for tau
            # https://doc.sagemath.org/html/en/reference/stats/sage/stats/distributions/discrete_gaussian_integer.html#sage.stats.distributions.discrete_gaussian_integer.DiscreteGaussianDistributionIntegerSampler.__init__
            self.__tau = 6
            self.__sigma = 8.5
        elif nist_security_level == 1:
            self.__nist_security_level = 1
            self.__n = 1024
            self.__B = 2 ** 19 - 1
            self.__h = 25
            self.__tau = 6
            self.__sigma = 8.5
        else:
            raise ValueError('The NIST security level must be 1 or 3.')

    @property
    def n(self):
        return self.__n

    @property
    def h(self):
        return self.__h

    @property
    def B(self):
        return self.__B

    @property
    def tau(self):
        return self.__tau

    @property
    def sigma(self):
        return self.__sigma

    @property
    def beta(self):
        """Maximum absolute value of an sc coefficient."""
        return self.h * (self.tau * self.sigma)

    @property
    def nist_security_level(self):
        return self.__nist_security_level

    @property
    def s_1_range(self):
        absolute_bound = ceil(self.tau * self.sigma)
        return range(-absolute_bound, absolute_bound + 1)

    @property
    def y_range(self):
        return range(-self.B, self.B + 1)

    @property
    def max_diff(self):
        return 2 * self.h * self.tau * self.sigma + self.B

    @property
    def dtype(self):
        return np.int32

    @property
    def threads(self):
        return self.__threads
