C
C  $Id: is.h,v 1.10 1997/08/22 15:22:51 balay Exp bsmith $;
C
C  Include file for Fortran use of the IS (index set) package in PETSc
C
#define IS                     integer
#define ISType                 integer
#define ISLocalToGlobalMapping integer
#define ISColoring             integer

      integer IS_GENERAL,IS_STRIDE, IS_BLOCK

      parameter (IS_GENERAL = 0, IS_STRIDE = 1, IS_BLOCK = 2)

C
C  End of Fortran include file for the IS package in PETSc

