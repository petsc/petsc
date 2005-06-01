
!
!  Include file for Fortran use of the DA (distributed array) package in PETSc
!
#if !defined (__PETSCDA_H)
#define __PETSCDA_H

#define DA PetscFortranAddr
#define DAPeriodicType PetscEnum
#define DAStencilType PetscEnum
#define DADirection PetscEnum
#define DMMG PetscFortranAddr
#define VecPack PetscFortranAddr
#define DM PetscFortranAddr

#define DALocalInfo integer
!
!   DA_LOCAL_INFO_SIZE is one large than the size incase the DA is larger than an integer (on 64 bit systems).
!   non-int fields are not accessiable from fortran.
!
#define DA_LOCAL_INFO_SIZE 22
#define DA_LOCAL_INFO_DIM 1
#define DA_LOCAL_INFO_DOF 2
#define DA_LOCAL_INFO_MX 4
#define DA_LOCAL_INFO_MY 5
#define DA_LOCAL_INFO_MZ 6
#define DA_LOCAL_INFO_XS 7
#define DA_LOCAL_INFO_YS 8
#define DA_LOCAL_INFO_ZS 9
#define DA_LOCAL_INFO_XM 10
#define DA_LOCAL_INFO_YM 11
#define DA_LOCAL_INFO_ZM 12
#define DA_LOCAL_INFO_GXS 13
#define DA_LOCAL_INFO_GYS 14
#define DA_LOCAL_INFO_GZS 15
#define DA_LOCAL_INFO_GXM 16
#define DA_LOCAL_INFO_GYM 17
#define DA_LOCAL_INFO_GZM 18

#define XG_RANGE in(DA_LOCAL_INFO_GXS)+1:in(DA_LOCAL_INFO_GXS)+in(DA_LOCAL_INFO_GXM)
#define YG_RANGE in(DA_LOCAL_INFO_GYS)+1:in(DA_LOCAL_INFO_GYS)+in(DA_LOCAL_INFO_GYM)
#define ZG_RANGE in(DA_LOCAL_INFO_GZS)+1:in(DA_LOCAL_INFO_GZS)+in(DA_LOCAL_INFO_GZM)
#define X_RANGE in(DA_LOCAL_INFO_XS)+1:in(DA_LOCAL_INFO_XS)+in(DA_LOCAL_INFO_XM)
#define Y_RANGE in(DA_LOCAL_INFO_YS)+1:in(DA_LOCAL_INFO_YS)+in(DA_LOCAL_INFO_YM)
#define Z_RANGE in(DA_LOCAL_INFO_ZS)+1:in(DA_LOCAL_INFO_ZS)+in(DA_LOCAL_INFO_ZM)

#define DAInterpolationType PetscEnum
#define DA_Q0 0
#define DA_Q1 1

#endif


#if !defined (PETSC_AVOID_DECLARATIONS)
!
!  Types of stencils
!
      PetscEnum DA_STENCIL_STAR,DA_STENCIL_BOX

      parameter (DA_STENCIL_STAR = 0,DA_STENCIL_BOX = 1)
!
!  Types of periodicity
!
      PetscEnum DA_NONPERIODIC,DA_XPERIODIC,DA_YPERIODIC,DA_XYPERIODIC
      PetscEnum DA_XYZPERIODIC,DA_XZPERIODIC,DA_YZPERIODIC,DA_ZPERIODIC

      parameter (DA_NONPERIODIC = 0,DA_XPERIODIC = 1,DA_YPERIODIC = 2)
      parameter (DA_XYPERIODIC = 3,DA_XYZPERIODIC = 4)
      parameter (DA_XZPERIODIC = 5,DA_YZPERIODIC = 6,DA_ZPERIODIC = 7)
!
! DA Directions
!
      PetscEnum DA_X,DA_Y,DA_Z

      parameter (DA_X = 0,DA_Y = 1,DA_Z = 2)
!
!  End of Fortran include file for the DA package in PETSc

#endif
