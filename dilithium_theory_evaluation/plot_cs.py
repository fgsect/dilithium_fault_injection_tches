import math

import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy.signal
from scipy.stats.distributions import rv_discrete, randint
from parameters import Parameters


class QteslaParameters:
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
        absolute_bound = math.ceil(self.tau * self.sigma)
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


def qtesla_s_coefficient(params: QteslaParameters, sigma=None, limit=None) -> rv_discrete:
    if sigma is None:
        sigma = params.sigma
    if limit is None:
        a = -params.beta
        b = params.beta
    else:
        a = -limit
        b = limit

    x = np.arange(a, b + 1, dtype=params.dtype)
    if not hasattr(qtesla_s_coefficient, "discrete_gaussian") or True:
        pmf = np.exp(- x ** 2 / sigma ** 2 / 2) / (sigma * np.sqrt(2 * np.pi))
        values = (x, pmf)
        qtesla_s_coefficient.discrete_gaussian = rv_discrete(a=a, b=b, values=values).freeze()

    return qtesla_s_coefficient.discrete_gaussian


def new_convolve(rv_1: rv_discrete, rv_2: rv_discrete, convolve_func=np.convolve) -> rv_discrete:
    pad_amount = np.abs(rv_1.a - rv_2.a)
    if rv_1.a >= rv_2.a:
        pmf_1 = np.concatenate((np.zeros(pad_amount), rv_1.pmf(np.arange(rv_1.a, rv_1.b + 1))))
        pmf_2 = rv_2.pmf(np.arange(rv_2.a, rv_2.b + 1))
    else:
        pmf_1 = rv_1.pmf(np.arange(rv_1.a, rv_1.b + 1))
        pmf_2 = np.concatenate((np.zeros(pad_amount), rv_2.pmf(np.arange(rv_2.a, rv_2.b + 1))))

    convolved_pmf_raw = convolve_func(pmf_1, pmf_2)
    convolved_pmf = convolved_pmf_raw[pad_amount:]

    new_a = rv_1.a + rv_2.a
    new_b = rv_1.b + rv_2.b

    convolved_x = np.arange(new_a, new_b + 1)
    assert np.shape(convolved_x.shape) == np.shape(convolved_pmf.shape)

    new_rv = rv_discrete(new_a, new_b, values=(np.arange(new_a, new_b + 1), convolved_pmf))

    return new_rv.freeze()


def plot_pmf(rv: rv_discrete, f_name: str, no_bar=False, x_lim=None):
    x = np.arange(rv.a, rv.b + 1)
    y = rv.pmf(x)

    if x_lim is None:
        t = np.max(x)
    else:
        t = x_lim

    if no_bar:
        plt.plot(x, y)
    else:
        plt.bar(x, y)
    plt.ylabel('probability')
    plt.xlabel('coefficient value')
    plt.xlim([-t, t])
    plt.savefig(f_name + '.pdf', bbox_inches='tight', transparent=True)
    plt.show()


def plot_pmf_dilithium() -> None:
    level = 3
    params = Parameters.get_nist_security_level(level)
    result = randint(params.s_1_range.start, params.s_1_range.stop)
    for _ in range(params.tau):
        result = new_convolve(result, randint(params.s_1_range.start, params.s_1_range.stop))
    plot_pmf(result, 'dilithium_l3_cs_plot')


def plot_pmf_bliss() -> None:
    n = 512

    delta_1 = .42
    delta_2 = .03

    d_1 = math.floor(delta_1 * n)
    d_2 = math.floor(delta_2 * n)

    pm1 = rv_discrete(-1, 1, values=([-1, 0, 1], [.5, 0, .5])).freeze()
    pm2 = rv_discrete(-2, 2, values=([-2, -1, 0, 1, 2], [.5, 0, 0, 0, .5])).freeze()

    result = pm1
    for _ in range(d_1 - 1):
        result = new_convolve(result, pm1)
    for _ in range(d_2):
        result = new_convolve(result, pm2)

    plot_pmf(result, 'bliss_l3_cs_plot')


def plot_pmf_qtesla() -> None:
    level = 1
    params = QteslaParameters(level)
    sigmas = numpy.full(params.h, params.sigma)
    sigmas_squared = sigmas ** 2
    sigmas_sum = np.sum(sigmas_squared)
    new_sigma = np.sqrt(sigmas_sum)
    print(new_sigma)
    #exit()
    x_path = f'qtesla_l{params.nist_security_level}_x.npy'
    y_path = f'qtesla_l{params.nist_security_level}_y.npy'
    try:
        x = np.load(x_path)
        y = np.load(y_path)
        result = rv_discrete(np.min(x), np.max(x), values=(x, y))
    except FileNotFoundError:
        result = qtesla_s_coefficient(params)
        for _ in range(params.h - 1):
            result = new_convolve(result, qtesla_s_coefficient(params), convolve_func=np.convolve)
            print('convolce step')
        print('convolve done')
        x = np.arange(result.a, result.b + 1)
        y = result.pmf(x)
        np.save(x_path, x)
        np.save(y_path, y)
    plot_pmf(qtesla_s_coefficient(params, sigma=new_sigma), 'qtesla_l3_cs_plot', x_lim=params.tau*new_sigma)


def main() -> None:
    plot_pmf_dilithium()
    plot_pmf_bliss()
    plot_pmf_qtesla()


if __name__ == '__main__':
    main()
