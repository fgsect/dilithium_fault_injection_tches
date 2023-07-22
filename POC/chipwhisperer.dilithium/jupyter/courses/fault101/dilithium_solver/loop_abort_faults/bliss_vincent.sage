from sage.stats.distributions.discrete_gaussian_integer \
import DiscreteGaussianDistributionIntegerSampler
import random as py_random
import numpy
import numpy.linalg
import tqdm
import solve_cs_gurobi

# BLISS -II parameters
q = 12289
n = 512

(delta1, delta2) = (0.3, 0)
sigma = 10
kappa = 23

R.<xx>= QuotientRing(ZZ[x], ZZ[x]. ideal(x ^ n+1))
Rq.<xxx>= QuotientRing(GF(q)[x], GF(q)[x].ideal(x ^ n+1))
sampler = DiscreteGaussianDistributionIntegerSampler(
    sigma=sigma, algorithm='uniform+table')


def s1gen():
    s1vec = [0]*n
    d1 = ceil(delta1*n)
    d2 = ceil(delta2*n)

    while d1 > 0:
        i = randint(0, n-1)
        if s1vec[i] == 0:
            s1vec[i] = (-1) ^ randint(0, 1)
            d1 -= 1

    while d2 > 0:
        i = randint(0, n-1)
        if s1vec[i] == 0:
            s1vec[i] = 2*(-1) ^ randint(0, 1)
            d2 -= 1

    return sum([s1vec[i]*xx^i for i in range(n)]), s1vec


def faultyz1gen(s1, d):
    # y1=sum([ sampler ()*xx^i for i in range(d)])
    # y1=sum([ sampler ()*xx^i for i in range(d)])
    y1 = sum([sampler()*xx ^ i for i in py_random.sample(list(range(n)), d)])
    print("y1", y1)
    # c is a random binary polynomial of weight kappa
    dc = kappa
    cvec = [0]*n
    while dc > 0:
        i = randint(0, n-1)
        if cvec[i] == 0:
            cvec[i] = 1
            dc -= 1
    c = sum([cvec[i]*xx ^ i for i in range(n)])
    z1 = y1+c*s1
    return (c, z1)

def get_matrix_from_faulty_signature(s1, d):
    (c, z1) = faultyz1gen(s1, d)
    A = numpy.zeros((n, n))
    b = numpy.zeros(n)
    for iprime in tqdm.tqdm(range(n)):
        b[iprime] = z1[iprime]
    # Now, create the following equation system
    # We know that z = y_i + \sum_{i} \sum_{j} (-1)^{floor(i+j/n)} s_i b_j x^{(i+j) mod n}
    # Which means
    # z_{iprime} = y_i + \sum_{i+j \mod n == iprime} (-1)^{floor(i+j/n)} s_i b_j
    for i in range(n):
        for j in range(n):
            iprime = (((i + j) % n))
            if ((((int(i) + int(j)) // n) % 2) == 0):
                sign = 1
            else:
                sign = -1
            A[iprime][j] += sign*c[i]
    for iprime in tqdm.tqdm(range(n)):
        continue #res = 0
        for j in range(n):
            res += A[iprime][j]*s1[j]
        #print("res", res)
        #print("iprime", b[iprime])
        #assert (res == b[iprime])
    return A,b

def faultattack_shuffled_y1(d):
    s1,s1_vec = s1gen()
    s1_lifted = s1_vec #s1.list()#.coefficients(sparse=False)
    #s1_lifted = s1.list()
    print("s1", len(s1_lifted))
    A, b = get_matrix_from_faulty_signature(s1, d)
    for _ in range(3):
        Atmp, btmp = get_matrix_from_faulty_signature(s1, d)
        A = numpy.vstack((A,Atmp))
        b = numpy.vstack((b,btmp))
    b = b.flatten()
    print("lenght b", b.shape)
    print("length A", A.shape)
    """shat_not_rounded = numpy.linalg.lstsq(A, b, rcond=None)[0]
    max_s1 = 1.0
    min_s1 = -1.0
    for i in range(n):
        if shat_not_rounded[i]>max_s1:
            shat_not_rounded[i]= max_s1
        elif shat_not_rounded[i]<min_s1:
            shat_not_rounded[i] = min_s1
    start_s = numpy.round(shat_not_rounded).reshape(n)#/shat_not_rounded.max()).reshape(n)
    print("s1",s1_lifted)
    print("shat not rounded", start_s)
    for i in range(n):
         if s1_lifted[i] != start_s[i]:
            print("unequal at pos ", i, s1_lifted[i] , "start_s", start_s[i], "shat not rounded", shat_not_rounded[i])
    """
    ret_s = solve_cs_gurobi.solve_max_rows_simplified(A, b, actual_s=s1_lifted)
    return
    for i in range(n):
        if s1_lifted[i] != ret_s[i]:
            print("unequal at pos ", i, s1_lifted[i] , "start_s", ret_s[i])
    #print("least-squares", numpy.round(ns/ns.max()).reshape(512))



faultattack_shuffled_y1(100)
