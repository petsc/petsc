C
C  $Id: is.h,v 1.8 1996/08/05 04:15:45 bsmith Exp bsmith $;
C
C  Include file for Fortran use of the IS (index set) package in PETSc
C
#define IS       integer
#define ISType   integer

      integer IS_GENERAL,IS_STRIDE, IS_BLOCK

      parameter (IS_GENERAL = 0, IS_STRIDE = 1, IS_BLOCK = 2)

C
C  End of Fortran include file for the IS package in PETSc

