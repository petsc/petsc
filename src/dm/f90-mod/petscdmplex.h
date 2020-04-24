!
!  Include file for Fortran use of the DMPlex package in PETSc
!
#include "petsc/finclude/petscdmplex.h"

!
! DMPlexCellRefinerType
!
      PetscEnum, parameter :: DM_REFINER_REGULAR = 0
      PetscEnum, parameter :: DM_REFINER_TO_BOX = 1
      PetscEnum, parameter :: DM_REFINER_TO_SIMPLEX = 2
!
! DMPlexInterpolatedFlag
!
      PetscEnum, parameter :: DMPLEX_INTERPOLATED_INVALID = -1
      PetscEnum, parameter :: DMPLEX_INTERPOLATED_NONE = 0
      PetscEnum, parameter :: DMPLEX_INTERPOLATED_PARTIAL = 1
      PetscEnum, parameter :: DMPLEX_INTERPOLATED_MIXED = 2
      PetscEnum, parameter :: DMPLEX_INTERPOLATED_FULL = 3
