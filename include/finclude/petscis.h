C
C  $Id: is.h,v 1.5 1996/02/12 20:28:12 bsmith Exp balay $;
C
C  Include file for Fortran use of the IS (index set) package in PETSc
C
#define IS             integer
#define IndexSetType   integer

      integer IS_SEQ,IS_STRIDE_SEQ

      parameter (IS_SEQ=0, IS_STRIDE_SEQ=2)

C
C  End of Fortran include file for the IS package in PETSc

