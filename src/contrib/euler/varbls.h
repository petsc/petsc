!
!  Local arrays, including ghost points, for the current
!  iterate, corresponding to the X vector in PETSc code.
!  Space is allocated in UserCreateEuler().
!
!  Note: The original code stored each component variable in
!  a separate array in the common block /varbls/  ... In the
!  current code, we reorder the variables and store them in
!  the array xx (which corresponds exactly to the local array
!  for the PETSc vector X).  This provides better cache locality
!  and alleviates the need for translation back and forth between
!  formats.
!
!  Uniprocessor array sizes:
!      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
!      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
!
!      double precision  r(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
!      double precision ru(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
!      double precision rv(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
!      double precision rw(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
! /*   double precision  e(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1) */
      double precision  p(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)

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

        double precision 
     &    xx(ndof,gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)

! ---------------------------------------------------------------
!
!  Local arrays, including ghost points, for the full potential
!  unknowns, corresponding to the X vector in PETSc code.
!  Space is allocated in UserCreateEuler().
!
!       double precision fp(gxsf1:gxefp1,gysf1:gyefp1,gzsf1:gzefp1)
!
!  Uniprocessor array sizes:
!      double precision FP(NI1,NJ1,NK1),RH(NI1,NJ1,NK1)

#define FP(i,j,k) xx(ndof,i,j,k)

#define fp(i,j,k) xx(ndof,i,j,k)

