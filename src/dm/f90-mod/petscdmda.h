!
!  Used by petscdmdamod.F90 to create Fortran module file
!
#include "petsc/finclude/petscdmda.h"

!
!  Types of stencils
!
      PetscEnum, parameter :: DMDA_STENCIL_STAR = 0
      PetscEnum, parameter :: DMDA_STENCIL_BOX = 1
!
! DMDAInterpolationType
!
      PetscEnum, parameter :: DMDA_Q0=0
      PetscEnum, parameter :: DMDA_Q1=1
!
!     DMDAElementType
!
      PetscEnum, parameter :: DMDA_ELEMENT_P1=0
      PetscEnum, parameter :: DMDA_ELEMENT_Q1=1
