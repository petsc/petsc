c
c  Parallel array sizes, including necessary ghost points, for the
c  Jacobian matrix blocks (all but main diagonal).  Space is allocated
c  in UserCreateEuler() if the flag -mat_assemble_directly is NOT specified.
c
c  These blocks are the default storage scheme used within the
c  Julianne code.  They can also be directly used by the PETSc
c  block diagonal matrix formats.  However, for the case of using
c  fully implicit boundary conditions, many diagonals contain
c  nonzeros instead of just these 7.  So, we prefer to form the
c  matrix directly.  Alternatively, we could consider a hybrid
c  matrix format.
c
c  Note: These dimensions MUST agree with those in the routine nd() !!
c
      double precision B1(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      double precision B2(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      double precision B3(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      double precision B4(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      double precision B5(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      double precision B6(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
c
c  Uniprocessor array sizes
c
c      COMMON /M/    B1(5,5,NI1,NJ1,NK1),B2(5,5,NI1,NJ1,NK1)
c      COMMON /M/    B3(5,5,NI1,NJ1,NK1),B4(5,5,NI1,NJ1,NK1)
c      COMMON /M/    B5(5,5,NI1,NJ1,NK1),B6(5,5,NI1,NJ1,NK1)
c
c  For diagonal terms, use the file
c
c  #include "diag.h"
c

