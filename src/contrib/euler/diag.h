c
c  Parallel array sizes for the main diagonal blocks of the
c  Jacobian matrix.  Space is allocated in UserCreateEuler().
c
      double precision  D(5,5,xsf1:xef01,ysf1:yef01,zsf1:zef01)
c      double precision  D(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
c
c  Uniprocessor array sizes:
c      COMMON /DIAG/ D(5,5,NI1,NJ1,NK1)
