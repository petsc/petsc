c
c  Local arrays, including ghost points, for the current
c  iterate, corresponding to the X vector in PETSc code.
c  Space is allocated in UserCreateEuler().
c
c  Note: The original code stored each component variable in
c  a separate array in the common block /varbls/  ... In the
c  current code, we reorder the variables and store them in
c  the array xx (which corresponds exactly to the local array
c  for the PETSc vector X).  This provides better cache locality
c  and alleviates the need for translation back and forth between
c  formats.
c
c  Uniprocessor array sizes:
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c
c      Double  r(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c      Double ru(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c      Double rv(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c      Double rw(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
c      Double  e(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
      Double  p(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)

#define R(i,j,k) xx(1,i,j,k)
#define RU(i,j,k) xx(2,i,j,k)
#define RV(i,j,k) xx(3,i,j,k)
#define RW(i,j,k) xx(4,i,j,k)
#define E(i,j,k) xx(5,i,j,k)

#define r(i,j,k) xx(1,i,j,k)
#define ru(i,j,k) xx(2,i,j,k)
#define rv(i,j,k) xx(3,i,j,k)
#define rw(i,j,k) xx(4,i,j,k)
#define e(i,j,k) xx(5,i,j,k)

        Double xx(5,gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)

