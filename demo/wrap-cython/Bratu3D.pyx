from petsc4py.PETSc cimport Vec,  PetscVec
from petsc4py.PETSc cimport Mat,  PetscMat
from petsc4py.PETSc cimport DA,   PetscDA
from petsc4py.PETSc cimport SNES, PetscSNES

from petsc4py.PETSc import Error

cdef extern from "Bratu3Dimpl.h":
    ctypedef struct Params:
        double lambda_
    int FormInitGuess(PetscDA da, PetscVec x, Params *p)
    int FormFunction (PetscDA da, PetscVec x, PetscVec F, Params *p)
    int FormJacobian (PetscDA da, PetscVec x, PetscMat J, Params *p)

def formInitGuess(Vec x, DA da, double lambda_):
    cdef int ierr
    cdef Params p = {"lambda_" : lambda_}
    ierr = FormInitGuess(da.da, x.vec, &p)
    if ierr != 0: raise Error(ierr)

def formFunction(SNES snes, Vec x, Vec f, DA da, double lambda_):
    cdef int ierr
    cdef Params p = {"lambda_" : lambda_}
    ierr = FormFunction(da.da, x.vec, f.vec, &p)
    if ierr != 0: raise Error(ierr)

def formJacobian(SNES snes, Vec x, Mat J, Mat P, DA da, double lambda_):
    cdef int ierr
    cdef Params p = {"lambda_" : lambda_}
    ierr = FormJacobian(da.da, x.vec, P.mat, &p)
    if ierr != 0: raise Error(ierr)
    if J != P: J.assemble() # for matrix-free operator
    return Mat.Structure.SAME_NONZERO_PATTERN
