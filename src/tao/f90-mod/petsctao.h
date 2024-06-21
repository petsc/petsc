!
!  Used by petsctaomod.F90 to create Fortran module file
!
#include "petsc/finclude/petsctao.h"

      type, extends(tPetscObject) :: tTao
      end type tTao
      Tao, parameter :: PETSC_NULL_TAO = tTao(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_TAO
#endif

      type, extends(tPetscObject) :: tTaoLineSearch
      end type tTaoLineSearch
      TaoLineSearch, parameter :: PETSC_NULL_TAO_LINESEARCH = tTaoLineSearch(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_TAO_LINESEARCH
#endif

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

      PetscEnum, parameter ::  TAO_SUBSET_SUBVEC = 0
      PetscEnum, parameter ::  TAO_SUBSET_MASK = 1
      PetscEnum, parameter ::  TAO_SUBSET_MATRIXFREE = 2
