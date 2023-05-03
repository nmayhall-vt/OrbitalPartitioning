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
    with open('orbitalpartitioning/tests/test_data/data_CrOCr.pickle', 'rb') as handle:
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
        (Of, Sf, Vf), (_, _, _) = orbitalpartitioning.spade_partitioning((Cdocc, Csing, Cvirt), Pf[fi], S)
        Cfrags.append(np.hstack((Of, Sf, Vf)))
        ndocc_f = Of.shape[1]
        init_fspace.append((ndocc_f+Sf.shape[1], ndocc_f))
        nmof = Of.shape[1] + Sf.shape[1] + Vf.shape[1]
        clusters.append(list(range(orb_index, orb_index+nmof)))
        orb_index += nmof



    # Orthogonalize Fragment orbitals
    Cfrags = orbitalpartitioning.sym_ortho(Cfrags, S)

    Cact = np.hstack(Cfrags)

    test1 = Cact.T @ S @ data["Cact"]
    assert(np.trace(np.abs(test1)) - 16 < 1e-8)
    assert(init_fspace == data["init_fspace"])
    assert(clusters == data["clusters"])

