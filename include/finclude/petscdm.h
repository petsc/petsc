
!
!  Include file for Fortran use of the DM (distributed array) package in PETSc
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
      PetscEnum DMDA_NONGHOSTED
      PetscEnum DMDA_XPERIODIC
      PetscEnum DMDA_YPERIODIC
      PetscEnum DMDA_XYPERIODIC
      PetscEnum DMDA_XYZPERIODIC
      PetscEnum DMDA_XZPERIODIC
      PetscEnum DMDA_YZPERIODIC
      PetscEnum DMDA_ZPERIODIC
      PetscEnum DMDA_XGHOSTED
      PetscEnum DMDA_YGHOSTED
      PetscEnum DMDA_ZGHOSTED
      PetscEnum DMDA_XYZGHOSTED

      parameter (DMDA_NONPERIODIC = Z'0')
      parameter (DMDA_NONGHOSTED = Z'0')
      parameter (DMDA_XPERIODIC = Z'3')
      parameter (DMDA_YPERIODIC = Z'C')
      parameter (DMDA_XYPERIODIC = Z'F')
      parameter (DMDA_XYZPERIODIC = Z'3F')
      parameter (DMDA_XZPERIODIC = Z'33')
      parameter (DMDA_YZPERIODIC = Z'3C')
      parameter (DMDA_ZPERIODIC = Z'30')
      parameter (DMDA_XGHOSTED = Z'1')
      parameter (DMDA_YGHOSTED = Z'4')
      parameter (DMDA_ZGHOSTED = Z'10')
      parameter (DMDA_XYZGHOSTED = Z'15')

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
!  End of Fortran include file for the DM package in PETSc

