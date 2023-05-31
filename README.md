OrbitalPartitioning
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/nmayhall-vt/orbitalpartitioning/workflows/CI/badge.svg)](https://github.com/nmayhall-vt/orbitalpartitioning/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/nmayhall-vt/OrbitalPartitioning/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/OrbitalPartitioning/branch/main)



A very simple package that contains a few functions that perform SVD-based orbital rotations. This doesn't depend on any particular electronic structure theory packages, only numpy/scipy. 

## Function: `svd_subspace_partitioning`
Find orbitals that most strongly overlap with the projector, `P`,  by doing rotations within each orbital block. This function will split a list of Orbital Spaces up into separate `fragment` and `environment` blocks, _while maintiaing the same number of fragment orbitals as specified by the projector_. 

For example, if we have 3 orbital blocks, say the occupied, singly, and virtual orbitals, 

    CF, CE  = svd_subspace_partitioning([Cocc, Csing, Cvirt], P, S)
    (Cocc_f, Csing_f,  Cvirt_f) = CF
    (Cocc_e, Csing_e,  Cvirt_e) = CE
we will get back the list of fragment orbitals from each space, and a list of environment orbitals from each space. 
This function above, keeps only the largest singular values across all subspaces, so that the number of columns in each of the `CF` blocks is equal to the number for fragment orbitals (i.e., the rank of the projector).

## Function: `svd_subspace_partitioning_orth`
Find orbitals that most strongly overlap with the list of **orthogonalized AOs** listed in `frag`,  by doing rotations within each orbital block. This function will split a list of Orbital Spaces up into separate `fragment` and `environment` blocks, _while maintiaing the same number of fragment orbitals as specified by the projector_. 

For example, if we have 3 orbital blocks, say the occupied, singly, and virtual orbitals, 

    CF, CE  = svd_subspace_partitioning([Cocc, Csing, Cvirt], frag, S)
    (Cocc_f, Csing_f,  Cvirt_f) = CF
    (Cocc_e, Csing_e,  Cvirt_e) = CE
we will get back the list of fragment orbitals from each space, and a list of environment orbitals from each space. 

## Function: `svd_subspace_partitioning_nonorth`
Find orbitals that most strongly overlap with the list of **non-orthogonalized AOs** listed in `frag`,  by doing rotations within each orbital block. This function will split a list of Orbital Spaces up into separate `fragment` and `environment` blocks, _while maintiaing the same number of fragment orbitals as specified by the projector_. 

For example, if we have 3 orbital blocks, say the occupied, singly, and virtual orbitals, 

    CF, CE  = svd_subspace_partitioning([Cocc, Csing, Cvirt], frag, S)
    (Cocc_f, Csing_f,  Cvirt_f) = CF
    (Cocc_e, Csing_e,  Cvirt_e) = CE
we will get back the list of fragment orbitals from each space, and a list of environment orbitals from each space. 

## Function: `sym_ortho`

Symmetrically orthogonalize list of MO coefficients. E.g., 


    [C1, C2, C3, ... ] = sym_ortho([C1, C2, C3, ...], S, thresh=1e-8):
    
where each `Cn` matrix is a set of MO vectors in the AO basis, $C_{\mu,p}$. 

## Function: `canonicalize`
Given an AO Fock matrix, rotate each orbital block in `orbital_blocks` to diagonalize F

    [C1, C2, C3, ... ] = canonicalize([C1, C2, C3, ...], F)

## Function: `extract_frontier_orbitals`
Given an AO Fock matrix, split each orbital block into 3 spaces, NDocc, NAct, Nvirt


    Cenv, Cact, Cvir = extract_frontier_orbitals([C1, C2, C3, ...], F, dims)
    (Cenv1, Cenv2, ...) = Cenv
    (Cact1, Cact2, ...) = Cact
    (Cvir1, Cvir2, ...) = Cvir
    
    `dims` = [(#env, #act, #vir), (#env, #act, #vir), ...]
    `F`: the fock matrix  


## Function: `spade_partitioning` (NYI)
In SPADE [(ref)](https://pubs.acs.org/doi/10.1021/acs.jctc.8b01112), the occupied space is partitioned into two smaller spaces:
1. the `fragment` space: 
the orbitals that most strongly overlaps with a user specified set of atoms (or rather their basis functions). This is computed by performing an SVD of the projection of the occupied orbitals onto the specified AOs. The `fragment` orbitals are thus the span of the projected AO's. In the original paper, we also truncated the number of orbitals we keep, by dividing at the largest gap in the singular values. 
2. the `environment` space: 
the remaining orbitals. This includes not only the null space of the projected occupied orbitals, but also any singular vectors discarded from the `fragment` space. 

example usage:

    C_frag, C_env = spade_partitioning(C_occ, P, S)

where:
- `C_occ` is a numpy matrix of the occupied MO coefficients, $C_{\mu,i}$
- `P` is a AO x nfrag projection matrix (really it's the span of the projection matrix) that defines the AO's to project onto, defining the fragment. For typical cases, this will just be selected columns of the $S^{1/2}_{\mu\nu}$ matrix, indicating that the occupied space is being projected onto the symetrically orthogonalized AOs. Keeping only columns of the identity matrix corresponds to projection onto the non-orthogonal AOs. 
- `S` is the AO x AO overlap matrix. 

## Function: `DMET_partitioning` (NYI)

---
### Copyright

Copyright (c) 2023, Nick Mayhall


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
