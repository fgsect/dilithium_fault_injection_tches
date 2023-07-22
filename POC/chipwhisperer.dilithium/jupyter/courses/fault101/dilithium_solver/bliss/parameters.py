#!/usr/bin/env python3
import numpy as np
from math import ceil


class Parameters:
    def __init__(self, parameter_set: int = 3, threads: int = 4):
        self.__threads = threads
        if parameter_set == 0:
            raise ValueError('Toy not (yet) implemented.')
        if not (parameter_set == 4 or parameter_set == 3 or parameter_set == 2 or parameter_set == 1):
            raise ValueError('Invalid parameter set.')
        if parameter_set == 4:
            self.__parameter_set = parameter_set
            self.__n = 512
            self.__delta_1 = 0.45
            self.__delta_2 = 0.06
            self.__kappa = 39
            self.__tau = 6
            self.__sigma = 271
        elif parameter_set == 3:
            self.__parameter_set = parameter_set
            self.__n = 512
            self.__delta_1 = 0.42
            self.__delta_2 = .03
            self.__kappa = 30
            # tau is chosen to be 6 as this is the default in sagemath
            # I do not think the BLISS authors mentioned an explicit value for tau
            # https://doc.sagemath.org/html/en/reference/stats/sage/stats/distributions/discrete_gaussian_integer.html#sage.stats.distributions.discrete_gaussian_integer.DiscreteGaussianDistributionIntegerSampler.__init__
            self.__tau = 6
            self.__sigma = 250
        elif parameter_set == 2:
            self.__parameter_set = parameter_set
            self.__n = 512
            self.__delta_1 = 0.3
            self.__delta_2 = 0
            self.__kappa = 23
            self.__tau = 6
            self.__sigma = 107
        elif parameter_set == 1:
            self.__parameter_set = parameter_set
            self.__n = 512
            self.__delta_1 = 0.3
            self.__delta_2 = 0
            self.__kappa = 23
            self.__tau = 6
            self.__sigma = 215

    @property
    def n(self):
        return self.__n

    @property
    def delta_1(self):
        return self.__delta_1

    @property
    def delta_2(self):
        return self.__delta_2

    @property
    def kappa(self):
        return self.__kappa

    @property
    def tau(self):
        return self.__tau

    @property
    def sigma(self):
        return self.__sigma

    @property
    def beta(self):
        how_often_two = min(self.kappa, self.num_plus_minus_two_coefficients)
        kappas_left_over = max(0, self.kappa - how_often_two)
        how_often_one = min(kappas_left_over, self.num_plus_minus_one_coefficients)
        return how_often_one * 1 + how_often_two * 2

    @property
    def num_plus_minus_one_coefficients(self):
        return ceil(self.delta_1 * self.n)

    @property
    def num_plus_minus_two_coefficients(self):
        return ceil(self.delta_2 * self.n)

    @property
    def num_zero_coefficients(self):
        return self.n - self.num_plus_minus_one_coefficients - self.num_plus_minus_two_coefficients

    @property
    def parameter_set(self):
        return self.__parameter_set

    @property
    def s_1_range(self):
        if min(self.delta_1, self.delta_2) > 0:
            return range(-2, 2 + 1)
        else:
            return range(-1, 1 + 1)

    @property
    def y_1_range(self):
        bound = self.tau * self.sigma
        return range(-bound, bound + 1)

    @property
    def max_diff(self):
        return 2 * self.beta + max(self.y_1_range)

    @property
    def dtype(self):
        return np.int32

    @property
    def threads(self):
        return self.__threads

    def __eq__(self, other):
        if type(other) is type(self) and other.parameter_set == self.parameter_set:
            return True
        return False

    def __hash__(self):
        return self.parameter_set
