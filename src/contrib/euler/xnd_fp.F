c ----------------------------------------------------------------
c  Full potential physics code initial condition
c ----------------------------------------------------------------

      SUBROUTINE IC_FP(xx,p,x,y,z)

c IC - Initializes the full potential component of the field to plug flow

#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "freq.h"
#include "ang.h"
#include "init.h"
#include "condi.h"
#include "coord.h"

c      COMMON /ANG/ ALPHA,BETA,PHI
c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /INIT/ RINIT,RUINIT,RVINIT,RWINIT,EINIT
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c      COMMON /CONDI/ ORDER,PSI,LIMIT
c      COMMON /FREQ/ IFREQ,ISGS
      double precision  conv, beta_dummy
      integer i,j,k,ifreq_dummy
c
c Reading would be redundant with IC for Euler if both models are 
c in use, but is necessary when running only the FP model.  This
c section MUST be skipped if already performed in the IC for Euler,
c or the angle "conversion" will be done twice!
c Many of the input parameters are not relevant for FP, but are
c kept as dummies for input file items.
c
      if (model .gt. MMFP) goto 40
      READ (15,100) CFL,FSMACH,ALPHA,BETA_dummy,PHI
      read (15,200) limit,psi,order,isgs,ifreq_dummy
c LIMIT chooses the limiter (see below)
c PSI blends discretization stencils for desired ORDER (see DEK for reference)
c ORDER is discretization order
c ISGS is the number of symmetric Gauss-Seidel sweeps per Newton step
c IFREQ determines how often the flux Jacobians are updated
      IF (ORDER.EQ.3.) PSI=1./3.
      IF (LIMIT.GT.1) PSI=0.
      IF (LIMIT.GT.1) ORDER=2.
C   LIMIT=0 (NO LIMITERS, 1st order), LIMIT=1 (MINMOD), LIMIT=2 (SUPERBEE)
C   LIMIT=3 (VAN LEER), LIMIT=4 (VAN ALBADA), LIMIT=5 (NO LIMITERS)
  100 FORMAT (8E10.0)
  200 FORMAT (I5,2E10.0,2i5)
      if (rank .eq. 0) then
        WRITE (6,20)
        WRITE (6,30) FSMACH,ALPHA,BETA,PHI,CFL,IFREQ,ISGS,NI,NJ,NK
   20   FORMAT (1H1,2X,'MACH NO.    ALPHA    BETA    PHI    CFL  IFREQ',
     * 3X,'ISGS    NI     NJ     NK')
   30   FORMAT (F10.4,F10.2,F8.2,F7.2,F7.1,2I7,I6,2I7,/)
        WRITE (6,21)
        WRITE (6,31) LIMIT,PSI,ORDER
   21   FORMAT (5X,'LIMIT       PSI   ORDER')
   31   FORMAT (I10,F10.4,F8.2,/)
      endif
      CONV=ACOS(-1.0d0)/180.
c      CONV=ACOS(PetscDoubleExp(-1.0,0))/180.
      ALPHA=ALPHA*CONV
      BETA=BETA*CONV
      PHI=PHI*CONV
c
   40 continue
c
c Input and angle conversion to radians is done.
c Define plug flow initial condition for potential, with zero circulation
c and constant of integration equal to zero (so PHI is zero at the origin
c of coordinates and along the line normal to the freestream flow through
c the origin).
c
      RUINIT=FSMACH*COS(ALPHA)*COS(BETA)
      RVINIT=FSMACH*(SIN(PHI)*SIN(ALPHA)*COS(BETA)
     1   -COS(PHI)*SIN(BETA))
      RWINIT=FSMACH*(SIN(ALPHA)*COS(BETA)
     1  +SIN(PHI)*COS(ALPHA)*SIN(BETA))
      do 2 k=gzsf1,gzefp1
      do 2 j=gysf1,gyefp1
      do 2 i=gxsf1,gxefp1
c      DO 2 K=1,NK1
c      DO 2 J=1,NJ1
c      DO 2 I=1,NI1
          FP(I,J,K) = 
     &    x(i,j,k)*RUINIT + y(i,j,k)*RVINIT + z(i,j,k)*RWINIT 
    2 CONTINUE
      RETURN
      END

