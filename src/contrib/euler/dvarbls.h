c
c  Parallel array sizes, including ghost points, for the
c  residual, corresponding to the F vector in PETSc code.
c  Space is allocated in UserCreateEuler().
c
      double precision  dr(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision dru(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision drv(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision drw(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision  de(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c
c  Uniprocessor array sizes:
c      COMMON /DVRBLS/ DR(NI1,NJ1,NK1),DRU(NI1,NJ1,NK1),DRV(NI1,NJ1,NK1)
c      COMMON /DVRBLS/ DRW(NI1,NJ1,NK1),DE(NI1,NJ1,NK1)


