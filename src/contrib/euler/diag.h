c
c  Parallel array sizes for the main diagonal blocks of the
c  Jacobian matrix.  Space is allocated in UserCreateEuler().
c
      double precision
     &  D(ndof_e,ndof_e,xsf2:xef01,ysf2:yef01,zsf2:zef01)
c
c  Uniprocessor array sizes:
c      COMMON /DIAG/ D(ndof_e,ndof_e,NI1,NJ1,NK1)
