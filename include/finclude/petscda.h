!
!  $Id: petscda.h,v 1.17 2000/09/25 17:55:27 balay Exp bsmith $;
!
!  Include file for Fortran use of the DA (distributed array) package in PETSc
!
#if !defined (__PETSCDA_H)
#define __PETSCDA_H

#define DA PetscFortranAddr
#define DAPeriodicType integer
#define DAStencilType integer
#define DADirection integer
#define DMMG        integer
#endif


#if !defined (PETSC_AVOID_DECLARATIONS)
!
!  Types of stencils
!
      integer DA_STENCIL_STAR,DA_STENCIL_BOX

      parameter (DA_STENCIL_STAR = 0,DA_STENCIL_BOX = 1)
!
!  Types of periodicity
!
      integer DA_NONPERIODIC,DA_XPERIODIC,DA_YPERIODIC,DA_XYPERIODIC
      integer DA_XYZPERIODIC,DA_XZPERIODIC,DA_YZPERIODIC,DA_ZPERIODIC

      parameter (DA_NONPERIODIC = 0,DA_XPERIODIC = 1,DA_YPERIODIC = 2)
      parameter (DA_XYPERIODIC = 3,DA_XYZPERIODIC = 4)
      parameter (DA_XZPERIODIC = 5,DA_YZPERIODIC = 6,DA_ZPERIODIC = 7)
!
! DA Directions
!
      integer DA_X,DA_Y,DA_Z

      parameter (DA_X = 0,DA_Y = 1,DA_Z = 2)
!
!  End of Fortran include file for the DA package in PETSc

#endif
