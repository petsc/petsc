!
!  Used by petscdmplexmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscdmplex.h"

      type, extends(tPetscObject) :: tDMPlexTransform
      end type tDMPlexTransform
      DMPlexTransform, parameter :: PETSC_NULL_DMPLEXTRANSFORM = tDMPlexTransform(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DMPLEXTRANSFORM
#endif
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

