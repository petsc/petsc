!
!  $Id: mg.h,v 1.4 1998/03/24 16:11:29 balay Exp balay $;
!
!  Include file for Fortran use of the MG preconditioner in PETSc
!
#if !defined (__MG_H)
#define __MG_H

#define MGType    integer

#endif
!
!
      integer MGMULTIPLICATIVE, MGADDITIVE, MGFULL, MGKASKADE
      parameter (MGMULTIPLICATIVE=0,MGADDITIVE=1,MGFULL=2,MGKASKADE=3)

!
!  Other defines
!
      integer MG_V_CYCLE, MG_W_CYCLE
      parameter (MG_V_CYCLE=1, MG_W_CYCLE=2)

!
!     End of Fortran include file for the  MG include file in PETSc
