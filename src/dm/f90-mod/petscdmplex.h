!
!  Used by petscdmplexmod.F90 to create Fortran module file
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

      type tDMPlexTransform
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tDMPlexTransform

      DMPlexTransform, parameter :: PETSC_NULL_DMPLEXTRANSFORM = tDMPlexTransform(0)
