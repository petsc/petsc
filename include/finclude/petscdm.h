
!
!  Include file for Fortran use of the DMDA (distributed array) package in PETSc
!
#include "finclude/petscdmdef.h"


!
!  Types of stencils
!
      PetscEnum DMDA_STENCIL_STAR
      PetscEnum DMDA_STENCIL_BOX

      parameter (DMDA_STENCIL_STAR = 0,DMDA_STENCIL_BOX = 1)
!
!  Types of periodicity
!
      PetscEnum DMDA_NONPERIODIC
      PetscEnum DMDA_XPERIODIC
      PetscEnum DMDA_YPERIODIC
      PetscEnum DMDA_XYPERIODIC
      PetscEnum DMDA_XYZPERIODIC
      PetscEnum DMDA_XZPERIODIC
      PetscEnum DMDA_YZPERIODIC
      PetscEnum DMDA_ZPERIODIC
      PetscEnum DMDA_XYZGHOSTED

      parameter (DMDA_NONPERIODIC = 0,DMDA_XPERIODIC = 1)
      parameter (DMDA_YPERIODIC = 2)
      parameter (DMDA_XYPERIODIC = 3,DMDA_XYZPERIODIC = 4)
      parameter (DMDA_XZPERIODIC = 5,DMDA_YZPERIODIC = 6)
      parameter (DMDA_ZPERIODIC = 7)
      parameter (DMDA_XYZGHOSTED = 8)

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
! DMDA Directions
!
      PetscEnum DMDA_X
      PetscEnum DMDA_Y
      PetscEnum DMDA_Z

      parameter (DMDA_X = 0,DMDA_Y = 1,DMDA_Z = 2)
!
!  End of Fortran include file for the DMDA package in PETSc

