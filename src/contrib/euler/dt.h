!
!  Parallel array sizes for the pseudo-transient array.  Space
!  is allocated in UserCreateEuler().
!
      double precision     DT(xsf1:xefp1,ysf1:yefp1,zsf1:zefp1)
!      double precision  dt(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
!
!  Uniprocessor array sizes:
!      COMMON /TSTEP/ DT(NI1,NJ1,NK1)
