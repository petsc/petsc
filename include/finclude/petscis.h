C
C  $Id: is.h,v 1.6 1996/04/15 23:29:48 balay Exp bsmith $;
C
C  Include file for Fortran use of the IS (index set) package in PETSc
C
#define IS       integer
#define ISType   integer

      integer IS_SEQ,IS_STRIDE_SEQ

      parameter (IS_SEQ=0, IS_STRIDE_SEQ=1)

C
C  End of Fortran include file for the IS package in PETSc

