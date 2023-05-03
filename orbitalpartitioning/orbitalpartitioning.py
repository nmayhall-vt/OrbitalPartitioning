import numpy as np
import scipy


def spade_partitioning(orbitals_blocks, Pv, S):
    """
    Find orbitals that most strongly overlap with the projector, P,  by doing rotations within each orbital block. 
    [C1, C2, C3] -> [(C1f, C2f, C3f), (C1e, C2e, C3e)]
    where C1f (C2f) and C1e (C2e) are the fragment orbitals in block 1 (2) and remainder orbitals in block 1 (2).

    Common scenarios would be 
        `orbital_blocks` = [Occ, Virt]
        or 
        `orbital_blocks` = [Occ, Sing, Virt]
    
    P[AO, frag]
    O[AO, occupied]
    U[AO, virtual]
    """

    nfrag = Pv.shape[1]
    nbas = S.shape[0]
    assert(Pv.shape[0] == nbas)
    nmo = 0
    for i in orbitals_blocks:
        assert(i.shape[0] == nbas)
        nmo += i.shape[1]


    print(" Partition %4i orbitals into a total of %4i orbitals" %(nmo, Pv.shape[1]))
    PS = Pv.T @ S @ Pv

    P = Pv @ np.linalg.inv(PS) @ Pv.T


    s = []
    Clist = []
    spaces = []
    Cf = []
    Ce = []
    for obi, ob in enumerate(orbitals_blocks):
        _,sob,Vob = np.linalg.svd(P @ S @ ob, full_matrices=True)
        s.extend(sob)
        Clist.append(ob @ Vob.T)
        spaces.extend([obi for i in range(ob.shape[1])])
        Cf.append(np.zeros((nbas, 0)))
        Ce.append(np.zeros((nbas, 0)))

    spaces = np.array(spaces)
    s = np.array(s)

    # Sort all the singular values
    perm = np.argsort(s)[::-1]
    s = s[perm]
    spaces = spaces[perm]

    Ctot = np.hstack(Clist)
    Ctot = Ctot[:,perm]    

    print(" %16s %12s %-12s" %("Index", "Sing. Val.", "Space"))
    for i in range(nfrag):
        print(" %16i %12.8f %12s*" %(i, s[i], spaces[i]))
        block = spaces[i]
        Cf[block] = np.hstack((Cf[block], Ctot[:,i:i+1]))

    for i in range(nfrag, nmo):
        if s[i] > 1e-6:
            print(" %16i %12.8f %12s" %(i, s[i], spaces[i]))
        block = spaces[i]
        Ce[block] = np.hstack((Ce[block], Ctot[:,i:i+1]))


    return Cf, Ce 


def spade_partitioning_rhf(O, U, Pv, S):
    """
    Find orbitals that most strongly overlap with the projector, P,  by doing O-O and V-V rotations. 
    [O,U] -> Of, Uf, Oe, Ue
    where Of (Uf) and Oe (Ue) are the Occupied (Unoccupied) orbitals of the fragment and environment, respectively.
    
    P[AO, frag]
    O[AO, occupied]
    U[AO, virtual]
    """


    assert(O.shape[0] == U.shape[0])
    assert(Pv.shape[0] == U.shape[0])
    assert(S.shape[0] == U.shape[0])
    
    nbas = O.shape[0]
    nmo = O.shape[1] + U.shape[1]

    print(" Partition %4i occupied and %4i virtual orbitals into a total of %4i orbitals" %(O.shape[1], U.shape[1], Pv.shape[1]))
    PS = Pv.T @ S @ Pv

    P = Pv @ np.linalg.inv(PS) @ Pv.T

    #print(np.linalg.det(P.T @ S @ P))
    # assert(np.isclose(np.abs(np.linalg.det(P.T @ S @ P)), 1.0))
    nfrag = Pv.shape[1]

    _,so,Vo = np.linalg.svd(P @ S @ O, full_matrices=True)
    _,su,Vu = np.linalg.svd(P @ S @ U, full_matrices=True)

    s = np.concatenate((so, su))
    inds = [] 
    for i in range(len(so)):
        inds.append(i+1)
    for i in range(len(su)):
        inds.append(-i-1)
    inds = np.array(inds)

    spaces = ["O" for i in range(O.shape[1])]
    spaces.extend(["V" for i in range(U.shape[1])])
    spaces = np.array(spaces)

    perm = np.argsort(s)[::-1]
    inds = inds[perm]
    s = s[perm]
    spaces = spaces[perm]

    Crot = np.hstack((O @ Vo.T, U @ Vu.T))
    Crot = Crot[:,perm]    

    # print(" %16s %12s %12s" %("Singular Value", "Occupied", "Virtual"))
    # for i in range(nfrag):
    #     print(" %16i %12.8f %12.8f" %(i, so[i], su[i]))


    Of = np.zeros((nbas, 0))
    Oe = np.zeros((nbas, 0))
    Uf = np.zeros((nbas, 0))
    Ue = np.zeros((nbas, 0))

    # print(" %16s %12s %12s" %("--", "--", "--"))
    print(" %16s %12s %-12s" %("Index", "Sing. Val.", "Space"))
    for i in range(nfrag):
        print(" %16i %12.8f %12s*" %(i, s[i], spaces[i]))
        if spaces[i] == "O":
            Of = np.hstack((Of, Crot[:,i:i+1]))
        elif spaces[i] == "V":
            Uf = np.hstack((Uf, Crot[:,i:i+1]))
        else:
            error("ArithmeticError")
    

    for i in range(nfrag, nmo):
        if s[i] > 1e-6:
            print(" %16i %12.8f %12s" %(i, s[i], spaces[i]))
        if spaces[i] == "O":
            Oe = np.hstack((Oe, Crot[:,i:i+1]))
        elif spaces[i] == "V":
            Ue = np.hstack((Ue, Crot[:,i:i+1]))
        else:
            error("ArithmeticError")


    return  Of, Uf, Oe, Ue


def sym_ortho(frags, S, thresh=1e-8):
    """
    Orthogonalize list of MO coefficients. 
    
    `frags` is a list of mo-coeff matrices, e.g., [C[ao,mo], C[ao, mo], ...]
    """
    Nbas = S.shape[1]
    
    inds = []
    Cnonorth = np.hstack(frags)
    shift = 0
    for f in frags:
        inds.append(list(range(shift, shift+f.shape[1])))
        shift += f.shape[1]
        
    
    Smo = Cnonorth.T @ S @ Cnonorth
    X = np.linalg.inv(scipy.linalg.sqrtm(Smo))
    # print(Cnonorth.shape, X.shape)
    Corth = Cnonorth @ X
    
    frags2 = []
    for f in inds:
        frags2.append(Corth[:,f])
    return frags2






if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
