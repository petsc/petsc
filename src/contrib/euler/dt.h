c
c  Parallel array sizes for the pseudo-transient array.  Space
c  is allocated in UserCreateEuler().
c
      double precision     DT(xsf1:xefp1,ysf1:yefp1,zsf1:zefp1)
c      double precision  dt(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c
c  Uniprocessor array sizes:
c      COMMON /TSTEP/ DT(NI1,NJ1,NK1)
