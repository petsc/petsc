!
!  Parallel array sizes for the main diagonal blocks of the
!  Jacobian matrix.  Space is allocated in UserCreateEuler().
!
      double precision                                                  &
     &  D(ndof_e,ndof_e,xsf2:xef01,ysf2:yef01,zsf2:zef01)
!
!  Uniprocessor array sizes:
!      COMMON /DIAG/ D(ndof_e,ndof_e,NI1,NJ1,NK1)
