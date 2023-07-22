#! /usr/bin/env sage

from parameters import Parameters
from signature import gen_y_np, gen_c_np, gen_s_1_np, calculate_z_np
import numpy as np
import numpy.typing as npt

def array_to_sage(a: npt.ArrayLike, params: Parameters):
    if np.shape(a) == (params.n,):
        return sum([coefficient * x ** i for i, coefficient in enumerate(a)])
    else:
        assert np.shape(a) == (params.l, params.n)
        return vector(sum([coefficient * x ** i for i, coefficient in enumerate(a[i, :])]) for i in range(params.l))


def main() -> None:
    print('Here we test if the signature calculation function in signature.py works correctly.')

    level = 3
    params = Parameters.get_nist_security_level(level)
    m = params.n * params.l - 10

    global x
    R_q.<x> = QuotientRing(GF(params.q)[x], GF(params.q)[x].ideal(x ** params.n + 1))

    y_np = gen_y_np(m, params)
    c_np = gen_c_np(params)
    s_1_np = gen_s_1_np(params)

    y = array_to_sage(y_np, params)
    c = array_to_sage(c_np, params)
    s_1 = array_to_sage(s_1_np, params)

    z_np = calculate_z_np(y_np, c_np, s_1_np, params)
    z = y + c * s_1

    z_check = array_to_sage(z_np, params)

    assert z == z_check

    print(z_np)
    print(z)

    print('Seccess!')


if __name__ == '__main__':
    main()
