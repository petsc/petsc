!
!  $Id: mg.h,v 1.7 1999/09/14 15:58:30 balay Exp bsmith $;
!
!  Include file for Fortran use of the MG preconditioner in PETSc
!
#if !defined (__MG_H)
#define __MG_H

#define MGType    integer

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)
!
!
      integer MGMULTIPLICATIVE,MGADDITIVE,MGFULL,MGKASKADE,MGCASCADE
      parameter (MGMULTIPLICATIVE=0,MGADDITIVE=1,MGFULL=2,MGKASKADE=3)
      parameter (MGCASCADE=3)

!
!  Other defines
!
      integer MG_V_CYCLE,MG_W_CYCLE
      parameter (MG_V_CYCLE=1,MG_W_CYCLE=2)

!
!     End of Fortran include file for the  MG include file in PETSc

#endif
