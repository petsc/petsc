
!
!  Include file for Fortran use of the DM (distributed array) package in PETSc
!
#include "petsc/finclude/petscdmda.h"

!
!  Types of stencils
!
      PetscEnum DMDA_STENCIL_STAR
      PetscEnum DMDA_STENCIL_BOX

      parameter (DMDA_STENCIL_STAR = 0,DMDA_STENCIL_BOX = 1)

!
! DMDAInterpolationType
!
      PetscEnum DMDA_Q0
      PetscEnum DMDA_Q1
      parameter (DMDA_Q0=0,DMDA_Q1=1)

!
!     DMDAElementType
!
      PetscEnum DMDA_ELEMENT_P1
      PetscEnum DMDA_ELEMENT_Q1
      parameter(DMDA_ELEMENT_P1=0,DMDA_ELEMENT_Q1=1)
!
!  End of Fortran include file for the DM package in PETSc
