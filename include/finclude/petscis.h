!
!  $Id: is.h,v 1.12 1997/11/13 19:17:12 balay Exp balay $;
!
!  Include file for Fortran use of the IS (index set) package in PETSc
!
#define IS                     integer
#define ISType                 integer
#define ISLocalToGlobalMapping integer
#define ISColoring             integer
#define ISLocalToGlobalMapping integer

      integer IS_GENERAL,IS_STRIDE, IS_BLOCK
      parameter (IS_GENERAL = 0, IS_STRIDE = 1, IS_BLOCK = 2)

      integer IS_GTOLM_MASK,IS_GTOLM_DROP 
      parameter (IS_GTOLM_MASK =0, IS_GTOLM_DROP = 1)

!
!  End of Fortran include file for the IS package in PETSc

