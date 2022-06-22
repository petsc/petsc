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
!
! DMPlexTPSType
!
      PetscEnum, parameter :: DMPLEX_TPS_SCHWARZ_P = 0
      PetscEnum, parameter :: DMPLEX_TPS_GYROID = 1

      type tDMPlexTransform
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tDMPlexTransform

      DMPlexTransform, parameter :: PETSC_NULL_DMPLEXTRANSFORM = tDMPlexTransform(0)
!
! DMPlexReorderDefaultFlag
!
      PetscEnum, parameter :: DMPLEX_REORDER_DEFAULT_NOTSET = -1
      PetscEnum, parameter :: DMPLEX_REORDER_DEFAULT_FALSE = 0
      PetscEnum, parameter :: DMPLEX_REORDER_DEFAULT_TRUE = 1
