C
C  $Id: pc.h,v 1.14 1996/04/16 03:53:03 balay Exp $;
C
C  Include file for Fortran use of the MG preconditioner in PETSc
C
#define MGType    integer

C
C  MGType
C
      integer MGMULTIPLICATIVE, MGADDITIVE, MGFULL, MGKASKADE
      parameter (MGMULTIPLICATIVE=0,MGADDITIVE=1,MGFULL=2,MGKASKADE=3)

C
C  Other defines
C
      integer MG_V_CYCLE, MG_W_CYCLE
      parameter (MG_V_CYCLE=1, MG_W_CYCLE=1)

C
C     End of Fortran include file for the  MG include file in PETSc
