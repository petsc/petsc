C
C  $Id: is.h,v 1.8 1996/08/05 04:15:45 bsmith Exp $;
C
C  Include file for Fortran use of the IS (index set) package in PETSc
C
#define IS       integer
#define ISType   integer

      integer IS_SEQ,IS_STRIDE_SEQ, IS_BLOCK_SEQ

      parameter (IS_SEQ = 0, IS_STRIDE_SEQ = 1, IS_BLOCK_SEQ = 2)

C
C  End of Fortran include file for the IS package in PETSc

