c ---------------------------------------------------------------
c
c  Local auxiliary arrays, including ghost points, for the full potential
c  unknowns.  Space is allocated in UserCreateEuler().
c
       double precision den(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
       double precision xvel(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
       double precision yvel(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
       double precision zvel(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c
c  Uniprocessor array sizes:
c      double precision FP(NI1,NJ1,NK1),RH(NI1,NJ1,NK1)
c      double precision XVEL(NI1,NJ1,NK1),YVEL(NI1,NJ1,NK1),ZVEL(NI1,NJ1,NK1)

