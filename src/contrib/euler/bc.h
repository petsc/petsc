c
c  Parallel array sizes, including ghost points, for implementing 
c  certain boundary conditions that require scattering data across
c  the processors.  Space is allocated in UserCreateEuler().
c
c  Note:  These extra arrays are a convenient means of handling the
c         parallel vector scatter data.  We could conserve a bit of
c         space by handling this differently. 
c
      double precision  r_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision ru_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision rv_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision rw_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision  e_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision  p_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c

