c
c  Parallel array sizes, including ghost points, for the
c  current iterate, corresponding to the X vector in PETSc code.
c  Space is allocated in UserCreateEuler().
c
      double precision  r(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision ru(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision rv(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision rw(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision  e(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision  p(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c
c  Uniprocessor array sizes:
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)

