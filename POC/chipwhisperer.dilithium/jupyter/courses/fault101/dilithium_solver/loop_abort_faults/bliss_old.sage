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

    return sum([s1vec[i]*xx^i for i in range(n)])


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
    #print()
    #print(z1[0])
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
    s1 = s1gen()
    #s1_lifted = s1.lift().coefficients(sparse=False)
    s1_lifted = s1.list()
    print("s1", len(s1_lifted))
    A, b = get_matrix_from_faulty_signature(s1, d)
    for _ in range(3):
        Atmp, btmp = get_matrix_from_faulty_signature(s1, d)
        A = numpy.vstack((A,Atmp))
        b = numpy.vstack((b,btmp))
    b = b.flatten()
    print("lenght b", b.shape)
    print("length A", A.shape)
    shat_not_rounded = numpy.linalg.lstsq(A, b, rcond=None)[0]
    start_s = numpy.round(shat_not_rounded).reshape(n)#/shat_not_rounded.max()).reshape(n)
    print("s1",s1_lifted)
    print("shat not rounded", start_s)
    for i in range(n):
         if s1_lifted[i] != start_s[i]:
            print("unequal at pos ", i, s1_lifted[i] , "start_s", start_s[i], "shat not rounded", shat_not_rounded[i])
    ret_s = solve_cs_gurobi.solve_min_number_rows_indicator_variables(A, b, start_s=start_s,shat_not_rounded=shat_not_rounded,actual_s=s1.lift().coefficients(sparse=False))
    for i in range(n):
        if s1_lifted[i] != ret_s[i]:
            print("unequal at pos ", i, s1_lifted[i] , "start_s", ret_s[i])
    #print("least-squares", numpy.round(ns/ns.max()).reshape(512))


def faultattack(d,e,bkz_size =25):
    s1=s1gen()
    (c,z1)=faultyz1gen(s1 ,d)

    try:
        cinv =1/Rq(c.lift())
    except ZeroDivisionError:
        print("c not invertible")
        return s1,c,z1,matrix(ZZ ,[])

    """
    65 Try to recover the first e coefficients of s1
    (of course , if we succeed , we should succeed for *all* sets of
    67 e coefficients of s1 , so we can recover the whole secret key).
    """
    print("Starting attack")
    t=cputime(subprocesses=True)
    latvec =[( cinv*xxx^i).lift().list()[:e] for i in range(d)]
    latvec =[( cinv*Rq(z1.lift())).lift().list()[:e]] + latvec
    latvec=latvec +[[0]*i + [q] + [0]*(e-i-1) for i in range(e)]
    M=matrix(ZZ ,latvec)
    M=M.augment(matrix(ZZ ,e+d+1,1,[2*q]+[0]*(e+d)))
    if bkz_size is None:
        M=M.LLL()
    else:
        M=M.BKZ(block_size=bkz_size)
    v=M[d+e]
    v=v*(2*q/v[-1])
    print ("Attack time:", cputime(subprocesses=True)-t)
    print ("Recovered vector:", v[:-1])
    print ("Truncated key:", s1.lift().list()[:e])
    
    return s1,c,z1,M

def faultattack_multiple(d,e,bkz_size=None ,tries =100):
    succ=0
    secs =0.0
    for _ in range(tries):
        s1=s1gen ()
        while True:
            (c,z1)=faultyz1gen(s1 ,d)
            try:
                cinv =1/Rq(c.lift())
            except ZeroDivisionError:
                print("*")
                sys.stdout.flush()
                continue
            break
        
        t=cputime(subprocesses=True)

        latvec =[( cinv*xxx^i).lift().list()[:e] for i in range(d)]
        latvec =[( cinv*Rq(z1.lift())).lift().list()[:e]] + latvec
        latvec=latvec +[[0]*i + [q] + [0]*(e-i-1) for i in range(e)]

        M=matrix(ZZ ,latvec)
        M=M.augment(matrix(ZZ ,e+d+1,1,[2*q]+[0]*(e+d)))
        if bkz_size is None:
            M=M.LLL()
        else:
            M=M.BKZ(block_size=bkz_size)
        """
        v=M[d+e]
        v=v*(2*q/v[-1])
        t=cputime(subprocesses=True)-t
        secs+= float(t)
        if v[: -1]. list()==s1.lift().list()[:e]:
        succ +=1
        print "+",
        else:
        print ".",
        sys.stdout.flush()

        print
        133 print "Success: %d/%d (%f%%)" % (succ ,tries ,100*RR(succ/tries))
        print "Avg CPU time:", secs/tries
        135 print "Avg CPU time (total vec):", secs/tries*ceil(n/e*tries/succ)
        137 def theoretical_lattice_size(d):
        r=delta1 +4*delta2
        139 u=0.5*log(r)/log(q)
        v=0.5*log(2*pi*exp(1)*r)/log(q)
        141 return RR(((d+1+u)/(1-v)) -1)
        143 # vim: ft=python
        """

faultattack_shuffled_y1(100)#)n)
