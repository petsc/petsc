c
c  Parallel array sizes, including ghost points, for implementing 
c  certain boundary conditions that require scattering data across
c  the processors.  Space is allocated in UserCreateEuler().
c
c  Note:  These extra arrays are a convenient means of handling the
c         parallel vector scatter data.  We could conserve a bit of
c         space by handling this differently. 
c
      Double  r_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      Double ru_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      Double rv_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      Double rw_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      Double  e_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      Double  p_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c

