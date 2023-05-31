"""
Unit and regression test for the orbitalpartitioning package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import orbitalpartitioning
import pickle
import numpy as np
import scipy
import scipy.linalg

def test_orbitalpartitioning_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "orbitalpartitioning" in sys.modules


def test1():
    with open('orbitalpartitioning/tests/data/data_CrOCr.pickle', 'rb') as handle:
        data = pickle.load(handle)




    Pf     = data["Pf"]      
    Cdocc  = data["Cdocc"]   
    Csing  = data["Csing"]   
    Cvirt  = data["Cvirt"]   
    S      = data["S"]       
    frags  = data["frags"]       
    init_fspace = []
    clusters = []
    Cfrags = []
    orb_index = 1
    for fi,f in enumerate(frags):
        print()
        print(" Fragment: ", f)
        (Of, Sf, Vf), (_, _, _) = orbitalpartitioning.svd_subspace_partitioning((Cdocc, Csing, Cvirt), Pf[fi], S)
        Cfrags.append(np.hstack((Of, Sf, Vf)))
        ndocc_f = Of.shape[1]
        init_fspace.append((ndocc_f+Sf.shape[1], ndocc_f))
        nmof = Of.shape[1] + Sf.shape[1] + Vf.shape[1]
        clusters.append(list(range(orb_index, orb_index+nmof)))
        orb_index += nmof


    print(" ---------------------------------------------------------------- ")
    Cfrags2 = []
    orb_index = 1
    for fi,f in enumerate(frags):
        print()
        print(" Fragment: ", f)
        (Of, Sf, Vf), (_, _, _) = orbitalpartitioning.svd_subspace_partitioning_orth((Cdocc, Csing, Cvirt), f, S)
        Cfrags2.append(np.hstack((Of, Sf, Vf)))
        
    ref = [ 1,1,1]    
    test = [np.linalg.det(Cfrags[fi].T @ S @ Cfrags2[fi]) for fi,f in enumerate(frags)] 

    for i in range(len(ref)):
        assert(abs(test[i])-abs(ref[i]) < 1e-12)      
    
    for fi,f in enumerate(frags):
        print(" Should be 1: ", np.linalg.det(Cfrags[fi].T @ S @ Cfrags2[fi]))
    print(" ---------------------------------------------------------------- ")

    print(" ---------------------------------------------------------------- ")
    Cfrags2 = []
    orb_index = 1
    for fi,f in enumerate(frags):
        print()
        print(" Fragment: ", f)
        (Of, Sf, Vf), (_, _, _) = orbitalpartitioning.svd_subspace_partitioning_nonorth((Cdocc, Csing, Cvirt), f, S)
        Cfrags2.append(np.hstack((Of, Sf, Vf)))
        
    ref = [ 0.9639412220052649, -0.9867599366347694, -0.9639404218744322]    
    test = [np.linalg.det(Cfrags[fi].T @ S @ Cfrags2[fi]) for fi,f in enumerate(frags)] 

    for i in range(len(ref)):
        assert(abs(test[i])-abs(ref[i]) < 1e-12) 

    for fi,f in enumerate(frags):
        print(" Should be 1: ", np.linalg.det(Cfrags[fi].T @ S @ Cfrags2[fi]))
    print(" ---------------------------------------------------------------- ")

    # Orthogonalize Fragment orbitals
    Cfrags = orbitalpartitioning.sym_ortho(Cfrags, S)

    Cact = np.hstack(Cfrags)

    test1 = Cact.T @ S @ data["Cact"]
    assert(np.trace(np.abs(test1)) - 16 < 1e-8)
    assert(init_fspace == data["init_fspace"])
    assert(clusters == data["clusters"])


    # canonicalize
    # let's just assume S is our fock matrix for now
    F = S
    Cfrags = orbitalpartitioning.canonicalize(Cfrags, F)
    [print(i.shape) for i in Cfrags]
    env, act, vir = orbitalpartitioning.extract_frontier_orbitals(Cfrags, F, [(1,4,1), (1,2,1), (1,4,1)])

    [print(i.shape) for i in env]
    [print(i.shape) for i in act]
    [print(i.shape) for i in vir]


def test2():
    with open('orbitalpartitioning/tests/data/data_CrOCr.pickle', 'rb') as handle:
        data = pickle.load(handle)

    Pf     = data["Pf"]      
    Cdocc  = data["Cdocc"]   
    Csing  = data["Csing"]   
    Cvirt  = data["Cvirt"]   
    S      = data["S"]       
    frags  = data["frags"]       
    init_fspace = []
    clusters = []
    Cfrags = []
    orb_index = 1
    
    orbitals, init_fspace, clusters = orbitalpartitioning.dmet_clustering(Cdocc, Cvirt, frags, S)
    
    Cfull = np.hstack((orbitals))
    assert(Cfull.shape[1] == round(np.trace(Cfull @ Cfull.T @ S)))

    Cfull = orbitals[0]
    assert(Cfull.shape[1] == round(np.trace(Cfull @ Cfull.T @ S)))

    tmp = [i.shape[1] for i in orbitals]
    ref = [57,12,8,12,271]
    print(tmp, ref)
    assert(tmp == ref)

if __name__ == "__main__":
    test1()
    test2()