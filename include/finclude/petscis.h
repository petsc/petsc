!
!  $Id: is.h,v 1.13 1998/03/24 16:11:37 balay Exp balay $;
!
!  Include file for Fortran use of the IS (index set) package in PETSc
!
#define IS                     PETScAddr
#define ISLocalToGlobalMapping PETScAddr
#define ISColoring             PETScAddr
#define ISType                 integer

      integer IS_GENERAL,IS_STRIDE, IS_BLOCK
      parameter (IS_GENERAL = 0, IS_STRIDE = 1, IS_BLOCK = 2)

      integer IS_GTOLM_MASK,IS_GTOLM_DROP 
      parameter (IS_GTOLM_MASK =0, IS_GTOLM_DROP = 1)

!
!  End of Fortran include file for the IS package in PETSc

