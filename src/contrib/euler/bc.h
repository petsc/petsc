c
c  Parallel array sizes, including ghost points, for implementing 
c  certain boundary conditions that require scattering data across
c  the processors.  Space is allocated in UserCreateEuler().
c
c  Note:  These extra arrays are a convenient means of handling the
c         parallel vector scatter data.  We could conserve a bit of
c         space by handling this differently. 
c
c      double precision  r_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c      double precision ru_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c      double precision rv_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c      double precision rw_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c      double precision  e_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      double precision  p_bc(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c

#define r_bc(i,j,k) xx_bc(1,i,j,k)
#define ru_bc(i,j,k) xx_bc(2,i,j,k)
#define rv_bc(i,j,k) xx_bc(3,i,j,k)
#define rw_bc(i,j,k) xx_bc(4,i,j,k)
#define e_bc(i,j,k) xx_bc(5,i,j,k)

#define R_bc(i,j,k) xx_bc(1,i,j,k)
#define RU_bc(i,j,k) xx_bc(2,i,j,k)
#define RV_bc(i,j,k) xx_bc(3,i,j,k)
#define RW_bc(i,j,k) xx_bc(4,i,j,k)
#define E_bc(i,j,k) xx_bc(5,i,j,k)

       double precision
     &   xx_bc(ndof,gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
