!
!  Include file for Fortran use of the DMPlex package in PETSc
!
#include "petsc/finclude/petscdmplex.h"

!
! DMPlexInterpolatedFlag
!
      PetscEnum DMPLEX_INTERPOLATED_INVALID
      PetscEnum DMPLEX_INTERPOLATED_NONE
      PetscEnum DMPLEX_INTERPOLATED_PARTIAL
      PetscEnum DMPLEX_INTERPOLATED_MIXED
      PetscEnum DMPLEX_INTERPOLATED_FULL

      parameter (DMPLEX_INTERPOLATED_INVALID = -1)
      parameter (DMPLEX_INTERPOLATED_NONE = 0)
      parameter (DMPLEX_INTERPOLATED_PARTIAL = 1)
      parameter (DMPLEX_INTERPOLATED_MIXED = 2)
      parameter (DMPLEX_INTERPOLATED_FULL = 3)
