!
!  Include file for Fortran use of the DMPlex package in PETSc
!
#include "petsc/finclude/petscdmplex.h"

!
! DMPlexInterpolatedFlag
!
      PetscEnum, parameter :: DMPLEX_INTERPOLATED_INVALID = -1
      PetscEnum, parameter :: DMPLEX_INTERPOLATED_NONE = 0
      PetscEnum, parameter :: DMPLEX_INTERPOLATED_PARTIAL = 1
      PetscEnum, parameter :: DMPLEX_INTERPOLATED_MIXED = 2
      PetscEnum, parameter :: DMPLEX_INTERPOLATED_FULL = 3
