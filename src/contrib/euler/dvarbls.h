c
c  Parallel array sizes, including ghost points, for the
c  residual, corresponding to the F vector in PETSc code.
c  Space is allocated in UserCreateEuler().
c
      Double  dr(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      Double dru(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      Double drv(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      Double drw(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      Double  de(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c
c  Uniprocessor array sizes:
c      COMMON /DVRBLS/ DR(NI1,NJ1,NK1),DRU(NI1,NJ1,NK1),DRV(NI1,NJ1,NK1)
c      COMMON /DVRBLS/ DRW(NI1,NJ1,NK1),DE(NI1,NJ1,NK1)


