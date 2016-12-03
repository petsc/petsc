
!
!  Include file for Fortran use of the TAO (Optimization) package in PETSc
!
#include "petsc/finclude/petsctao.h"

      PetscEnum TAO_CONVERGED_GATOL
      PetscEnum TAO_CONVERGED_GRTOL
      PetscEnum TAO_CONVERGED_GTTOL
      PetscEnum TAO_CONVERGED_STEPTOL
      PetscEnum TAO_CONVERGED_MINF
      PetscEnum TAO_CONVERGED_USER
      PetscEnum TAO_DIVERGED_MAXITS
      PetscEnum TAO_DIVERGED_NAN
      PetscEnum TAO_DIVERGED_MAXFCN
      PetscEnum TAO_DIVERGED_LS_FAILURE
      PetscEnum TAO_DIVERGED_TR_REDUCTION
      PetscEnum TAO_DIVERGED_USER
      PetscEnum TAO_CONTINUE_ITERATING

      parameter ( TAO_CONVERGED_GATOL = 3)
      parameter ( TAO_CONVERGED_GRTOL = 4)
      parameter ( TAO_CONVERGED_GTTOL = 5)
      parameter ( TAO_CONVERGED_STEPTOL = 6)
      parameter ( TAO_CONVERGED_MINF = 7)
      parameter ( TAO_CONVERGED_USER = 8)
      parameter ( TAO_DIVERGED_MAXITS = -2)
      parameter ( TAO_DIVERGED_NAN = -4)
      parameter ( TAO_DIVERGED_MAXFCN = -5)
      parameter ( TAO_DIVERGED_LS_FAILURE = -6)
      parameter ( TAO_DIVERGED_TR_REDUCTION = -7)
      parameter ( TAO_DIVERGED_USER = -8)
      parameter ( TAO_CONTINUE_ITERATING = 0)
