C
C  $Id: is.h,v 1.11 1997/10/12 19:57:38 bsmith Exp balay $;
C
C  Include file for Fortran use of the IS (index set) package in PETSc
C
#define IS                     integer
#define ISType                 integer
#define ISLocalToGlobalMapping integer
#define ISColoring             integer
#define ISLocalToGlobalMapping integer

      integer IS_GENERAL,IS_STRIDE, IS_BLOCK
      parameter (IS_GENERAL = 0, IS_STRIDE = 1, IS_BLOCK = 2)

      integer IS_GTOLM_MASK,IS_GTOLM_DROP 
      parameter (IS_GTOLM_MASK =0, IS_GTOLM_DROP = 1)

C
C  End of Fortran include file for the IS package in PETSc

