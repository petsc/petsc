!
!  $Id: mg.h,v 1.3 1996/04/16 04:55:14 bsmith Exp balay $;
!
!  Include file for Fortran use of the MG preconditioner in PETSc
!
#define MGType    integer
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
