!
!  Parallel array sizes, including necessary ghost points, for the
!  Jacobian matrix blocks (all but main diagonal).  Space is allocated
!  in UserCreateEuler() if the flag -mat_assemble_directly is NOT specified.
!
!  These blocks are the default storage scheme used within the
!  Julianne code.  They can also be directly used by the PETSc
!  block diagonal matrix formats.  However, for the case of using
!  fully implicit boundary conditions, many diagonals contain
!  nonzeros instead of just these 7.  So, we prefer to form the
!  matrix directly.  Alternatively, we could consider a hybrid
!  matrix format.
!
!  Note: These dimensions MUST agree with those in the routine nd() !!
!
      double precision
     & B1(ndof,ndof,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1),
     & B2(ndof,ndof,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1),
     & B3(ndof,ndof,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1),
     & B4(ndof,ndof,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1),
     & B5(ndof,ndof,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1),
     & B6(ndof,ndof,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
!
!  Uniprocessor array sizes
!
!      COMMON /M/    B1(ndof,ndof,NI1,NJ1,NK1),B2(ndof,ndof,NI1,NJ1,NK1)
!      COMMON /M/    B3(ndof,ndof,NI1,NJ1,NK1),B4(ndof,ndof,NI1,NJ1,NK1)
!      COMMON /M/    B5(ndof,ndof,NI1,NJ1,NK1),B6(ndof,ndof,NI1,NJ1,NK1)
!
!  For diagonal terms, use the file
!
!  #include "diag.h"
!

