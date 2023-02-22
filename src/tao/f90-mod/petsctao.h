!
!  Used by petsctaomod.F90 to create Fortran module file
!
#include "petsc/finclude/petsctao.h"

      type tTao
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type
      Tao, parameter :: PETSC_NULL_TAO = tTao(0)

      PetscEnum, parameter ::  TAO_CONVERGED_GATOL = 3
      PetscEnum, parameter ::  TAO_CONVERGED_GRTOL = 4
      PetscEnum, parameter ::  TAO_CONVERGED_GTTOL = 5
      PetscEnum, parameter ::  TAO_CONVERGED_STEPTOL = 6
      PetscEnum, parameter ::  TAO_CONVERGED_MINF = 7
      PetscEnum, parameter ::  TAO_CONVERGED_USER = 8
      PetscEnum, parameter ::  TAO_DIVERGED_MAXITS = -2
      PetscEnum, parameter ::  TAO_DIVERGED_NAN = -4
      PetscEnum, parameter ::  TAO_DIVERGED_MAXFCN = -5
      PetscEnum, parameter ::  TAO_DIVERGED_LS_FAILURE = -6
      PetscEnum, parameter ::  TAO_DIVERGED_TR_REDUCTION = -7
      PetscEnum, parameter ::  TAO_DIVERGED_USER = -8
      PetscEnum, parameter ::  TAO_CONTINUE_ITERATING = 0
