C
C This file contains routines that are used in both the Julianne solver 
C and the PETSc solver.
C
c ************************************************************************
c Commented in lower case by D. E. Keyes, 11 March 94
c
c To run, do :
c x.nd < nd.in > nd.out &
c
c Be sure to compile with double precision switched on in the FORTRAN
c compiler, or else the numerical differencing parameter will be too
c small for any significance in the Jacobian.  To run in single precision,
c set eps to 1.e-3.
c
c The default grid is the ONERA M6 wing coarse grid (49x9x9) stored in
c m6c.  To run the fine grid case, the declaration statements must be
c increased.  Almost all of the parameters are hardcoded into the driver.
c File input from nd.in is read in routine IC, which also initializes
c the flow field to a plug flow, ignoring the wing.  This creates a large
c initial residual near the wing. 
c
c Time is used as a continuation parameter.  The time step array is set
c in routine EIGENV, based on a local CFL number.
c
c The unknowns are density, vector momentum, and internal energy.  Pressure
c is obtained through the equation of state.  Each grid point of the
c structured, unstaggered grid hosts five principal unknowns, plus
c pressure and geometric data.
c
c A more advanced nonlinear solver will advance CFL to high values (e.g.,
c in the thousands or beyond), which causes this method to approach
c the standard Newton method asymptotically.  However, it appears that boundary
c conditions are updated explicitly after each solve -- another area we
c wish to address.
C************************************************************************
      integer function JULIANNE( time_init, jsolve, app, cfl_init,
     &                    rtol, eps, b1, b2, b3, b4, b5, b6, d, dt,
     &                    r, ru, rv, rw, e, p,
     &                    dr, dru, drv, drw, de, 
     &                    br, bl, be, sadai, sadaj, sadak,
     &                    aix, ajx, akx, aiy, ajy, aky, aiz, ajz, akz,
     &                    x, y, z, f1, g1, h1,
     &                    sp, sm, sp1, sp2, sm1, sm2, beta1, ifreq1)

#include "param.h"
#include "matrixb.h"
#include "diag.h"
#include "dt.h"
#include "varbls.h"
#include "dvarbls.h"
#include "consts.h"
#include "wing.h"
#include "coord.h"
#include "freq.h"
#include "es.h"
#include "sada.h"
#include "avctrs.h"
#include "fv.h"
#include "sigma.h"
#include "ang.h"

C   NUMERICAL SOLUTION OF THE THREE-DIMENSIONAL TIME-DEPENDENT EULER
C   EQUATIONS USING FINITE-VOLUME DISCRETIZATION AND A FIRST, SECOND,
C   OR THIRD ORDER (SPATIAL) IMPLICIT SCHEME TO SOLVE A ROE AVERAGED
C   WAVE-SPLIT FORM OF THE EQUATIONS WITH LOCAL OR MINIMUM TIME
C   STEPPING ON STATIONARY GRIDS.
C   FOR PROBLEMS CONTACT THE OKLAHOMA CHEROKEE:
C   DAVE WHITFIELD, DEPARTMENT OF AEROSPACE ENGINEERING, MISSISSIPPI
C   STATE UNIVERSITY, BOX 6176, MISSISSIPPI STATE, MS 39762 (601-325-7704)
C   THIS VERSION OF RED COON IS FOR A WING.  IT USES NO SSD.
C   IT HAS ROE ON BOTH SIDES OF THE EQUATION.  IT TAKES A NUMERICAL
C   DERIVATIVE OF THE FIRST ORDER ROE FLUXES FOR THE JACOBIAN.
C   THE SOLVER IS A BLOCK GAUSS-SEIDEL ITERATION SCHEME.
C   FIRST GOT THIS TO RUN ON AUGUST 7, 1990, JULIANN`S BIRTHDAY.
C
C   `WHITE MAN BUILD BIG FIRE, SIT WAY BACK.
C    INDIAN BUILD LITTLE FIRE, SIT UP CLOSE.`
C
c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /COORD/ X(NI,NJ,NK),Y(NI,NJ,NK),Z(NI,NJ,NK)
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /FREQ/ IFREQ,ISGS

      Double  cfl_init,rtol,beta1,eps,time_init
      integer jsolve,app,nb,nt,ierr,ifreq1
#if defined(ACTIVATE_JULIANNE)
      integer m,converge,epsi
#endif
      julianne    = 0
#if !defined(ACTIVATE_JULIANNE)
      if (jsolve .eq. 1) then
         if (rank .eq. 0) 
     &    write(6,*) 'JULIANNE: Solver is not compiled. Edit makefile',
     & ' to define ACTIVATE_JULIANNE; then recompile xnd.F and xjul.F'
         julianne = -1
         return
      endif
#endif

      OPEN (15,FILE='nd.in',STATUS='OLD')
      NCYC = 0
      GAM  = PetscDoubleExp(1.4,0)
      RGAM = one/GAM
      GM1  = GAM-one
      RGM1 = one/GM1
      GS2  = GAM*p5
c Default angle of attack = 3.06 degrees
      if (beta1 .gt. 5.0 .or. beta1 .lt. -1.0) then
         beta = PetscDoubleExp(-3.06,0)
      else 
         beta = -beta1
      endif
C Default frequency of forming Jacobian is every 10 iterations
      if (ifreq1 .ge. 1) then
         ifreq = ifreq1
      else
         ifreq = 10
      endif
c Read input and set up initial conditions
      CALL IC(r,ru,rv,rw,e,p)
      cfl_init = cfl
      READ (15,100) NB,NT
      close(15)
  100 FORMAT (2I5)
c Read mesh elsewhere now, to facilitate the parallel case
c where we need only the local portion.
c      READ (2,200) ITL,ITU,ILE,KTIP
c      READ (2,250) X,Y,Z
c      close(2)
c  200 FORMAT (4I3)
c  250 FORMAT (8F10.5)

c      call printgrids(x,y,z)

c Set parallel parameters to use in these loops:
c    for i=itl+1,itu    ->  for i=itlp1,itup
c    for k=2,ktip       ->  for k=zsf2,ktipp
c    ilep
c
      itl1 = itl+1
      if (itl1 .le. xsf2) then
        if (itu .le. xef01) then
          itlp1 = xsf2
        else
c         eliminate loop for this processor
          itlp1 = ni1+1
        endif
      else 
        if (itl1 .le. xef01) then
          itlp1 = itl1
        else
c         eliminate loop for this processor
          itlp1 = ni1+1
        endif
      endif
      if (itu .le. xef01) then
        if (itlp1 .eq. ni1+1) then
c         eliminate loop for this processor
          itup = 0
        else
          itup = itu
        endif
      else
        if (itl1 .le. xef01) then
          itup = xef01
        else
c         eliminate loop for this processor
          itup = 0
        endif
      endif
      if (zsf2 .le. ktip .and. zef01 .ge. ktip) then
         ktipp = ktip
      else if (ktip .lt. zsf2) then
c        eliminate loop for this processor
         ktipp = 0
      else
         ktipp = zef01
      endif
      ilep = ile
      if (printg .eq. 1) then
         call PetscSequentialPhaseBegin(comm,1,ierr)
         write(6,*) 'rank,itl,itlp1,itu,itup,ile,ilep,ktip,ktipp = ',
     &              rank,itl,itlp1,itu,itup,ile,ilep,ktip,ktipp
#ifdef PARCH_rs6000
         call FLUSH_(6)
#else
         call flush(6)
#endif
         call PetscSequentialPhaseEnd(comm,1,ierr)
      endif

c Set up mesh metrics, once and for all
      CALL METRIC(sadai,sadaj,sadak,
     &            aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,x,y,z)


c Apply boundary conditions only for original solver!
c For SNES version, boundary conditions are handled elsewhere.
#if defined(ACTIVATE_JULIANNE)
      if (jsolve .eq. 1) 
     &   CALL BC_UNI(r,ru,rv,rw,e,p,sadai,sadaj,sadak,
     &               aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
      epsi = one/eps
#endif
      if (rank .eq. 0) WRITE (6,300) NB,NT
  300 FORMAT(1H0,I20,' PRINTOUTS EVERY ',I5,' CYCLES')
c Loop over blocks (just one block, here) and pseudo-time steps
c     DO 3 L=1,NB
#if defined(ACTIVATE_JULIANNE)
      if (jsolve .ne. 1) then
#endif
         if (rank .eq. 0) write(6,*) 'Calling SNES driver'
         if (no_output .eq. 0) then
            OPEN (29,FILE='snes.m')
            write(29,*) 'zsnes = ['
         endif
c        Note:  call pvar at conclusion of SNES solver.
#if defined(ACTIVATE_JULIANNE)
      else
         if (rank .eq. 0) write(6,*) 'Calling original solver'
c         if (no_output .eq. 0) then
            OPEN (29,FILE='julianne.m')
            write(29,*) 'zjul = ['
c         endif
         DO 2 M=1,NT
c At each step, solve a Newton problem, and update the boundary conditions
         CALL STEP(time_init,eps,epsi,rtol,converge,
     &             b1,b2,b3,b4,b5,b6,d,dt,
     &             r,ru,rv,rw,e,p,
     &             dr,dru,drv,drw,de,
     &             br,bl,be,sadai,sadaj,sadak,
     &             aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &             f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
         if (converge .eq. 1) goto 20
    2    CONTINUE
   20    continue
c Calculate the physical quanties of interest
         CALL PVAR(r,ru,rv,rw,e,p,
     &             aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,x,y,z)
      endif
      if (jsolve .eq. 1 .and. no_output .eq. 0) call cleanup
#endif
      julianne = 0
c   3 CONTINUE
      return
      END

C************************************************************************

      subroutine printgrids(x,y,z)

#include "param.h"
#include "coord.h"
#include "wing.h"

      integer i,j,k,id

      j=2
      DO 8 K=1,NK1
      DO 8 I=1,NI1
      IF (K.GT.KTIP .or.
     &    I.LE.ITL.OR.I.GT.ITU) then
         ID=NI+1-I
         write(6,*) 'k,i,id,x(i,j,k),x(id,j,k)',
     &       k,i,id,x(i,j,k),x(id,j,k)
      endif
 8    continue
      return
      end

C************************************************************************

      SUBROUTINE IC(r,ru,rv,rw,e,p)
c IC reads case data and initializes the flow field to plug flow
#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "freq.h"
#include "ang.h"
#include "init.h"
#include "condi.h"

c      COMMON /ANG/ ALPHA,BETA,PHI
c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /INIT/ RINIT,RUINIT,RVINIT,RWINIT,EINIT
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c      COMMON /CONDI/ ORDER,PSI,LIMIT
c      COMMON /FREQ/ IFREQ,ISGS
      Double  conv, beta_dummy
      integer i,j,k,ifreq_dummy

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
C   LIMIT=0 (NO LIMITERS), LIMIT=1 (MINMOD), LIMIT=2 (SUPERBEE)
C   LIMIT=3 (VAN LEER)
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
      CONV=ACOS(PetscDoubleExp(-1.0,0))/180.
      ALPHA=ALPHA*CONV
      BETA=BETA*CONV
      PHI=PHI*CONV
      RINIT=one
      RUINIT=FSMACH*COS(ALPHA)*COS(BETA)
      RVINIT=FSMACH*(SIN(PHI)*SIN(ALPHA)*COS(BETA)
     1   -COS(PHI)*SIN(BETA))
      RWINIT=FSMACH*(SIN(ALPHA)*COS(BETA)
     1  +SIN(PHI)*COS(ALPHA)*SIN(BETA))
       EINIT=one/(GAM*GM1)+p5*FSMACH**2
      PINIT=GM1*(EINIT-p5*FSMACH**2)
      do 2 k=gzsf1,gzefp1
      do 2 j=gysf1,gyefp1
      do 2 i=gxsf1,gxefp1
c      DO 2 K=1,NK1
c      DO 2 J=1,NJ1
c      DO 2 I=1,NI1
       R(I,J,K)=one
      RU(I,J,K)=RUINIT
      RV(I,J,K)=RVINIT
      RW(I,J,K)=RWINIT
       E(I,J,K)= EINIT
       P(I,J,K)= PINIT
    2 CONTINUE
      RETURN
      END

C************************************************************************

      integer function bcpart_j1(r,ru,rv,rw,e,p,
     &                   r_bc,ru_bc,rv_bc,rw_bc,e_bc,p_bc,
     &                   sadai,sadaj,sadak,
     &                   aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)

c
c Applies boundary conditions for j=1 plane
c
#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "init.h"
#include "bc.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /INIT/ RINIT,RUINIT,RVINIT,RWINIT,EINIT
      Double  ada,qda,ff,p0,r0,c0,pb,u,v,w,rqdq
      integer i,k
c      integer id

      bcpart_j1 = 0

      if (ysf1 .eq. 1) then
        do 2 k=gzsf2,gzef01
        do 2 i=gxsf2,gxef01
c        DO 2 K=2,NK
c        DO 2 I=2,NI
        IF (K.GT.KTIP) GO TO 4
        IF (I.LE.ITL.OR.I.GT.ITU) GO TO 4
        IF (NCYC.GT.10) GO TO 15
        ADA=SADAJ(I,1,K)**2
        QDA=RU(I,2,K)*AJX(I,1,K)+RV(I,2,K)*AJY(I,1,K)
     *     +RW(I,2,K)*AJZ(I,1,K)
        FF=two*QDA/ADA
        RU(I,1,K)=RU(I,2,K)-FF*AJX(I,1,K)
        RV(I,1,K)=RV(I,2,K)-FF*AJY(I,1,K)
        RW(I,1,K)=RW(I,2,K)-FF*AJZ(I,1,K)
        P(I,1,K)=P(I,2,K)
        R(I,1,K)=R(I,2,K)
        E(I,1,K)=E(I,2,K)
        GO TO 2
   15   CONTINUE
C   THESE ARE CHARACTERISTIC VARIABLE IMPERMEABLE WALL B.C.s
        QDA=(RU(I,2,K)*AJX(I,1,K)+RV(I,2,K)*AJY(I,1,K)
     *     +RW(I,2,K)*AJZ(I,1,K))/R(I,2,K)
        P0=p5*(P(I,1,K)+P(I,2,K))
        R0=p5*(R(I,1,K)+R(I,2,K))
        C0=SQRT(GAM*P0/R0)
        PB=P(I,2,K)-R0*C0*QDA/SADAJ(I,1,K)
        P(I,1,K)=2.*PB-P(I,2,K)
        R(I,1,K)=R(I,2,K)+2.*(PB-P(I,2,K))/C0**2
        FF=two*QDA/SADAJ(I,1,K)**2
        U=RU(I,2,K)/R(I,2,K)-FF*AJX(I,1,K)
        V=RV(I,2,K)/R(I,2,K)-FF*AJY(I,1,K)
        W=RW(I,2,K)/R(I,2,K)-FF*AJZ(I,1,K)
        RU(I,1,K)=U*R(I,1,K)
        RV(I,1,K)=V*R(I,1,K)
        RW(I,1,K)=W*R(I,1,K)
        RQDQ=p5*(RU(I,1,K)**2+RV(I,1,K)**2+RW(I,1,K)**2)/R(I,1,K)
        E(I,1,K)=P(I,1,K)*RGM1+RQDQ
        GO TO 2
    4   CONTINUE

        R(I,1,K)=R_bc(I,2,K)
        P(I,1,K)=P_bc(I,2,K)
        E(I,1,K)=E_bc(I,2,K)
        RU(I,1,K)=RU_bc(I,2,K)
        RV(I,1,K)=RV_bc(I,2,K)
        RW(I,1,K)=RW_bc(I,2,K)

c        ID=NI+1-I
c            write(20,100) k,i,id,R_bc(I,2,K),Ru_bc(I,2,K),
c     &            rv_bc(i,2,k),rw_bc(i,2,k),e_bc(i,2,k),p_bc(i,2,k)
c            write(21,100) k,i,id,R(ID,2,K),Ru(ID,2,K),
c     &            rv(id,2,k),rw(id,2,k),e(id,2,k),p(id,2,k)
c   20    continue
c  100 format(i3,i3,i3,6e16.8)

    2   CONTINUE
      endif

      RETURN
      END

C************************************************************************

      integer function BC(r,ru,rv,rw,e,p,sadai,sadaj,sadak,
     &                      aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
c
c Apply boundary conditions
#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "init.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /INIT/ RINIT,RUINIT,RVINIT,RWINIT,EINIT
      Double  ada,qda,ff,p0,r0,c0,pb,u,v,w,r0c0,ub
      Double  u2,v2,w2,con,vb,wb,rb
      integer i,j,k

c   Now do boundary conditions for j=1 side in bcpart_j1().
c   See bc_uni() for uniprocessor case.

      bc = 0
      if (yef01 .eq. nj) then
        do 20 k=gzsf2,gzef01
        do 20 i=gxsf2,gxef01
c        DO 20 K=2,NK
c        DO 20 I=2,NI
C   THESE ARE CHARACTERISTIC VARIABLE SUBSONIC INFLOW B.C.s
        R0=p5*(R(I,NJ,K)+R(I,NJ1,K))
        P0=p5*(P(I,NJ,K)+P(I,NJ1,K))
        C0=SQRT(GAM*P0/R0)
        R0C0=R0*C0
        U2=RU(I,NJ,K)/R(I,NJ,K)
        V2=RV(I,NJ,K)/R(I,NJ,K)
        W2=RW(I,NJ,K)/R(I,NJ,K)
        PB=p5*(PINIT+P(I,NJ,K)-R0C0/SADAJ(I,NJ,K)*(AJX(I,NJ,K)*(RUINIT-
     *   U2)+AJY(I,NJ,K)*(RVINIT-V2)+AJZ(I,NJ,K)*(RWINIT-W2)))
        RB=RINIT+(PB-PINIT)/C0**2
        CON=-(PINIT-PB)/(R0C0*SADAJ(I,NJ,K))
        UB=RUINIT+AJX(I,NJ,K)*CON
        VB=RVINIT+AJY(I,NJ,K)*CON
        WB=RWINIT+AJZ(I,NJ,K)*CON
        P(I,NJ1,K)=two*PB-P(I,NJ,K)
        R(I,NJ1,K)=two*RB-R(I,NJ,K)
        U=2.*UB-U2
        V=2.*VB-V2
        W=2.*WB-W2
        RU(I,NJ1,K)=U*R(I,NJ1,K)
        RV(I,NJ1,K)=V*R(I,NJ1,K)
        RW(I,NJ1,K)=W*R(I,NJ1,K)
        E(I,NJ1,K)=P(I,NJ1,K)*RGM1+p5*R(I,NJ1,K)*(U**2+V**2+W**2)
   20   CONTINUE
      endif

      if (zsf1 .eq. 1) then
        do 6 j=gysf2,gyef01
        do 6 i=gxsf2,gxef01
c        DO 6 J=2,NJ
c        DO 6 I=2,NI
        ADA=AKX(I,J,1)**2+AKY(I,J,1)**2+AKZ(I,J,1)**2
        QDA=RU(I,J,2)*AKX(I,J,1)+RV(I,J,2)*AKY(I,J,1)
     *     +RW(I,J,2)*AKZ(I,J,1)
        R(I,J,1)=R(I,J,2)
        E(I,J,1)=E(I,J,2)
        P(I,J,1)=P(I,J,2)
        FF=two*QDA/ADA
        RU(I,J,1)=RU(I,J,2)-FF*AKX(I,J,1)
        RV(I,J,1)=RV(I,J,2)-FF*AKY(I,J,1)
        RW(I,J,1)=RW(I,J,2)-FF*AKZ(I,J,1)
    6   CONTINUE
      endif

      if (zef01 .eq. nk) then
        do 7 j=gysf2,gyef01
        do 7 i=gxsf2,gxef01
c        DO 7 J=2,NJ
c        DO 7 I=2,NI
        ADA=AKX(I,J,NK)**2+AKY(I,J,NK)**2+AKZ(I,J,NK)**2
        QDA=RU(I,J,NK)*AKX(I,J,NK)+RV(I,J,NK)*AKY(I,J,NK)
     *     +RW(I,J,NK)*AKZ(I,J,NK)
        R(I,J,NK1)=R(I,J,NK)
        E(I,J,NK1)=E(I,J,NK)
        P(I,J,NK1)=P(I,J,NK)
        FF=two*QDA/ADA
        RU(I,J,NK1)=RU(I,J,NK)-FF*AKX(I,J,NK)
        RV(I,J,NK1)=RV(I,J,NK)-FF*AKY(I,J,NK)
        RW(I,J,NK1)=RW(I,J,NK)-FF*AKZ(I,J,NK)
    7   CONTINUE
      endif

      if (xef01 .eq. ni) then
        do 8 k=gzsf2,gzef01
        do 8 j=gysf2,gyef01
c        DO 8 K=2,NK
c        DO 8 J=2,NJ
C   THESE ARE CHARACTERISTIC VARIABLE SUBSONIC OUTFLOW B.C.s
         R0=p5*(R(NI1,J,K)+R(NI,J,K))
        C0=SQRT(GAM*PINIT/R0)
        R(NI1,J,K)=R(NI,J,K)+two*(PINIT-P(NI,J,K))/C0**2
        CON=-two*(PINIT-P(NI,J,K))/(SADAI(NI,J,K)*R0*C0)
        U=RU(NI,J,K)/R(NI,J,K)+AIX(NI,J,K)*CON
        V=RV(NI,J,K)/R(NI,J,K)+AIY(NI,J,K)*CON
        W=RW(NI,J,K)/R(NI,J,K)+AIZ(NI,J,K)*CON
        RU(NI1,J,K)=U*R(NI1,J,K)
        RV(NI1,J,K)=V*R(NI1,J,K)
        RW(NI1,J,K)=W*R(NI1,J,K)
        E(NI1,J,K)=PINIT*RGM1+p5*R(NI1,J,K)*(U**2+V**2+W**2)
    8   CONTINUE
      endif
      if (xsf1 .eq. 1) then
        do 9 k=gzsf2,gzef01
        do 9 j=gysf2,gyef01
        R0=p5*(R(1,J,K)+R(2,J,K))
        C0=SQRT(GAM*PINIT/R0)
        R(1,J,K)=R(2,J,K)+2.*(PINIT-P(2,J,K))/C0**2
        CON=two*(PINIT-P(2,J,K))/(SADAI(1,J,K)*R0*C0)
        U=RU(2,J,K)/R(2,J,K)+AIX(1,J,K)*CON
        V=RV(2,J,K)/R(2,J,K)+AIY(1,J,K)*CON
        W=RW(2,J,K)/R(2,J,K)+AIZ(1,J,K)*CON
        RU(1,J,K)=U*R(1,J,K)
        RV(1,J,K)=V*R(1,J,K)
        RW(1,J,K)=W*R(1,J,K)
        E(1,J,K)=PINIT*RGM1+p5*R(1,J,K)*(U**2+V**2+W**2)
c    8   CONTINUE
    9   CONTINUE
      endif

      RETURN
      END

C************************************************************************

      SUBROUTINE EIGENV(dt,r,ru,rv,rw,e,p,sadai,sadaj,sadak,
     &                  aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
c
c Forms diagonal blocks for eigendecomposition.
c
c Parallel version: Forms local dt(), using ghost points for 
c AIX, AIY, SADAI, etc.
c
#include "param.h"
#include "dt.h"
#include "varbls.h"
#include "consts.h"
#include "avctrs.h"
#include "sada.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
c      COMMON /TSTEP/ DT(NI,NJ,NK)
      integer i,j,k
      Double  c,aixa,aiya,aiza,ajxa,ajya,ajza,akxa,akya,akza
      Double  ei,ej,ek,conu,conv,conw

      do 2 k=zsf2,zef01
      do 2 j=ysf2,yef01
      do 2 i=xsf2,xef01
c      DO 2 K=2,NK
c      DO 2 J=2,NJ
c      DO 2 I=2,NI
      C=SQRT(GAM*P(I,J,K)/R(I,J,K))
      AIXA=AIX(I,J,K)+AIX(I-1,J,K)
      AIYA=AIY(I,J,K)+AIY(I-1,J,K)
      AIZA=AIZ(I,J,K)+AIZ(I-1,J,K)
      CONU=(RU(I,J,K)*AIXA+RV(I,J,K)*AIYA+RW(I,J,K)*AIZA)/R(I,J,K)
      EI=ABS(CONU)+C*(SADAI(I,J,K)+SADAI(I-1,J,K))
      AJXA=AJX(I,J,K)+AJX(I,J-1,K)
      AJYA=AJY(I,J,K)+AJY(I,J-1,K)
      AJZA=AJZ(I,J,K)+AJZ(I,J-1,K)
      CONV=(RU(I,J,K)*AJXA+RV(I,J,K)*AJYA+RW(I,J,K)*AJZA)/R(I,J,K)
      EJ=ABS(CONV)+C*(SADAJ(I,J,K)+SADAJ(I,J-1,K))
      AKXA=AKX(I,J,K)+AKX(I,J,K-1)
      AKYA=AKY(I,J,K)+AKY(I,J,K-1)
      AKZA=AKZ(I,J,K)+AKZ(I,J,K-1)
      CONW=(RU(I,J,K)*AKXA+RV(I,J,K)*AKYA+RW(I,J,K)*AKZA)/R(I,J,K)
      EK=ABS(CONW)+C*(SADAK(I,J,K)+SADAK(I,J,K-1))
      DT(I,J,K)=CFL*two/(EI+EJ+EK)
    2 CONTINUE
      RETURN
      END

C************************************************************************

      SUBROUTINE METRIC(sadai,sadaj,sadak,
     &                  aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,x,y,z)
c
c Computes mesh metrics:
c     aix, aiy, aiz, ajx, ajy, ajz, akx, aky, akz
c     sadai, sadaj, sadak
c
c Parallel version: for now, form complete global mesh metrics on each processor
c
#include "param.h"
#include "coord.h"
#include "avctrs.h"
#include "sada.h"

c      COMMON /COORD/ X(NI,NJ,NK),Y(NI,NJ,NK),Z(NI,NJ,NK)
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
      integer i,j,k
      Double  dx1,dx2,dy1,dy2,dz1,dz2

      DO 2 J=gysf2,gyef01
      DO 2 K=gzsf2,gzef01
      DO 2 I=gxsf1,gxef01
c      DO 2 J=2,NJ
c      DO 2 K=2,NK
c      DO 2 I=1,NI
      DX1=X(I,J,K)-X(I,J-1,K-1)
      DY1=Y(I,J,K)-Y(I,J-1,K-1)
      DZ1=Z(I,J,K)-Z(I,J-1,K-1)
      DX2=X(I,J-1,K)-X(I,J,K-1)
      DY2=Y(I,J-1,K)-Y(I,J,K-1)
      DZ2=Z(I,J-1,K)-Z(I,J,K-1)
      AIX(I,J,K)=(DY1*DZ2-DZ1*DY2)*p5
      AIY(I,J,K)=(DZ1*DX2-DX1*DZ2)*p5
      AIZ(I,J,K)=(DX1*DY2-DY1*DX2)*p5
      SADAI(I,J,K)=SQRT(AIX(I,J,K)**2+AIY(I,J,K)**2+AIZ(I,J,K)**2)
    2 CONTINUE
      DO 3 I=gxsf2,gxef01
      DO 3 K=gzsf2,gzef01
      DO 3 J=gysf1,gyef01
c      DO 3 I=2,NI
c      DO 3 K=2,NK
c      DO 3 J=1,NJ
      DX1=X(I,J,K)-X(I-1,J,K-1)
      DY1=Y(I,J,K)-Y(I-1,J,K-1)
      DZ1=Z(I,J,K)-Z(I-1,J,K-1)
      DX2=X(I,J,K-1)-X(I-1,J,K)
      DY2=Y(I,J,K-1)-Y(I-1,J,K)
      DZ2=Z(I,J,K-1)-Z(I-1,J,K)
      AJX(I,J,K)=(DY1*DZ2-DZ1*DY2)*p5
      AJY(I,J,K)=(DZ1*DX2-DX1*DZ2)*p5
      AJZ(I,J,K)=(DX1*DY2-DY1*DX2)*p5
      SADAJ(I,J,K)=SQRT(AJX(I,J,K)**2+AJY(I,J,K)**2+AJZ(I,J,K)**2)
    3 CONTINUE
      DO 4 I=gxsf2,gxef01
      DO 4 J=gysf2,gyef01
      DO 4 K=gzsf1,gzef01
c      DO 4 I=2,NI
c      DO 4 J=2,NJ
c      DO 4 K=1,NK
      DX1=X(I,J,K)-X(I-1,J-1,K)
      DY1=Y(I,J,K)-Y(I-1,J-1,K)
      DZ1=Z(I,J,K)-Z(I-1,J-1,K)
      DX2=X(I-1,J,K)-X(I,J-1,K)
      DY2=Y(I-1,J,K)-Y(I,J-1,K)
      DZ2=Z(I-1,J,K)-Z(I,J-1,K)
      AKX(I,J,K)=(DY1*DZ2-DZ1*DY2)*p5
      AKY(I,J,K)=(DZ1*DX2-DX1*DZ2)*p5
      AKZ(I,J,K)=(DX1*DY2-DY1*DX2)*p5
      SADAK(I,J,K)=SQRT(AKX(I,J,K)**2+AKY(I,J,K)**2+AKZ(I,J,K)**2)
    4 CONTINUE
      RETURN
      END

C************************************************************************

      SUBROUTINE RLVECS(QER,QEL,I1,J1,K1,IB,JB,KB,AX,AY,AZ,SADA,
     &                   r,ru,rv,rw,e,p,br,bl,be)
#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "es.h"

C   RLVECS COMPUTES THE RIGHT AND LEFT EIGENVECTORS (STORED IN BR AND
C   BL, RESPECTIVELY) AND THE PLUS AND MINUS EIGENVALUES (STORED IN
C   A PORTION OF BE), ALL WITH ROE AVERGING.
c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /ES/ BR(5,5,NI,NJ,NK),BL(5,5,NI,NJ,NK),BE(2,5,NI,NJ,NK)
c
c   Note: sada must have same dimensions as sadai,sadaj,sadak in sada.h
      Double  SADA(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
c   Note: ax,ay,az must have same dimensions as arrays in avctrs.h
      Double  ax(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
      Double  ay(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
      Double  az(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)

      Double  AXT(D_NI),AYT(D_NI),AZT(D_NI)
      Double  Q1(D_NI),Q2(D_NI),Q3(D_NI),Q4(D_NI),Q6(D_NI),
     +                 RQRQ(D_NI),TT(D_NI),RC(D_NI),C(D_NI)
      Double  qer(5),qel(5)
      Double  a,b,c1,c2,c3,phi,rr2,e1,e4,e5,csad,hroe
      Double  rl,rul,rvl,rwl,el,rr,rur,rvr,rwr,er,ra
      Double  rra1,ur,vr,wr,ul,vl,wl,pr,pl,hr,hl
      integer i,j,k,I1,J1,K1,IB,JB,KB,kpk1,jpj1,ipi1

      if (xsf1 .ne. 1) then
c        ib = gxsf1w
        ib = gxsf1
      endif
      if (ysf1 .ne. 1) then
        jb = gysf1
c        jb = gysf1w
      endif
      if (zsf1 .ne. 1) then
        kb = gzsf1
c        kb = gzsf1w
      endif

      DO 2 K=KB,gzefw
c      DO 2 K=KB,NK
      KPK1=K+K1
      DO 2 J=JB,gyefw
c      DO 2 J=JB,NJ
      JPJ1=J+J1
      DO 10 I=IB,gxefw
c      DO 10 I=IB,NI
      AXT(I)=zero
      AYT(I)=zero
      AZT(I)=zero
      IF (SADA(I,J,K).EQ.0.) GO TO 10
      AXT(I)=AX(I,J,K)/SADA(I,J,K)
      AYT(I)=AY(I,J,K)/SADA(I,J,K)
      AZT(I)=AZ(I,J,K)/SADA(I,J,K)
   10 CONTINUE
      DO 22 I=IB,gxefw
c      DO 22 I=IB,NI
      IPI1=I+I1
      rl=r(i,j,k)+qel(1)
      rul=ru(i,j,k)+qel(2)
      rvl=rv(i,j,k)+qel(3)
      rwl=rw(i,j,k)+qel(4)
      el=e(i,j,k)+qel(5)
      rr=r(ipi1,jpj1,kpk1)+qer(1)
      rur=ru(ipi1,jpj1,kpk1)+qer(2)
      rvr=rv(ipi1,jpj1,kpk1)+qer(3)
      rwr=rw(ipi1,jpj1,kpk1)+qer(4)
      er=e(ipi1,jpj1,kpk1)+qer(5)
      RA=SQRT(rl/rr)
      RRA1=1./(1.+RA)
      ur=rur/rr
      vr=rvr/rr
      wr=rwr/rr
      ul=rul/rl
      vl=rvl/rl
      wl=rwl/rl
      Q1(I)=rr*RA
      Q2(I)=(ur+ul*RA)*RRA1
      Q3(I)=(vr+vl*RA)*RRA1
      Q4(I)=(wr+wl*RA)*RRA1
      pr=gm1*(er-p5*rr*(ur**2+vr**2+wr**2))
      pl=gm1*(el-p5*rl*(ul**2+vl**2+wl**2))
      hl=p5*(ul**2+vl**2+wl**2)+gam*rgm1*pl/rl
      hr=p5*(ur**2+vr**2+wr**2)+gam*rgm1*pr/rr
      HROE=(hr+hl*ra)*rra1
      RQRQ(I)=Q2(I)**2+Q3(I)**2+Q4(I)**2
      C(I)=SQRT((HROE-p5*RQRQ(I))*GM1)
      Q6(I)=one/Q1(I)
      TT(I)=Q2(I)*AXT(I)+Q3(I)*AYT(I)+Q4(I)*AZT(I)
      RC(I)=Q1(I)*C(I)
      CSAD=C(I)*SADA(I,J,K)
      E1=TT(I)*SADA(I,J,K)
      E4=E1+CSAD
      E5=E1-CSAD
      BE(1,1,I,J,K)=p5*(E1+ABS(E1))
      BE(1,2,I,J,K)=BE(1,1,I,J,K)
      BE(1,3,I,J,K)=BE(1,1,I,J,K)
      BE(1,4,I,J,K)=p5*(E4+ABS(E4))
      BE(1,5,I,J,K)=p5*(E5+ABS(E5))
      BE(2,1,I,J,K)=p5*(E1-ABS(E1))
      BE(2,2,I,J,K)=BE(2,1,I,J,K)
      BE(2,3,I,J,K)=BE(2,1,I,J,K)
      BE(2,4,I,J,K)=p5*(E4-ABS(E4))
      BE(2,5,I,J,K)=p5*(E5-ABS(E5))
   22 CONTINUE
      RR2=one/SQRT(two)
      DO 23 I=IB,gxefw
c      DO 23 I=IB,NI
      A=Q1(I)*RR2/C(I)
      C1=p5*RQRQ(I)
      C2=C(I)*C(I)*RGM1
      BR(1,1,I,J,K)=AXT(I)
      BR(1,2,I,J,K)=AYT(I)
      BR(1,3,I,J,K)=AZT(I)
      BR(1,4,I,J,K)=A
      BR(1,5,I,J,K)=A
      BR(2,1,I,J,K)=Q2(I)*AXT(I)
      BR(2,2,I,J,K)=Q2(I)*AYT(I)-Q1(I)*AZT(I)
      BR(2,3,I,J,K)=Q2(I)*AZT(I)+Q1(I)*AYT(I)
      BR(2,4,I,J,K)=A*(Q2(I)+C(I)*AXT(I))
      BR(2,5,I,J,K)=A*(Q2(I)-C(I)*AXT(I))
      BR(3,1,I,J,K)=Q3(I)*AXT(I)+Q1(I)*AZT(I)
      BR(3,2,I,J,K)=Q3(I)*AYT(I)
      BR(3,3,I,J,K)=Q3(I)*AZT(I)-Q1(I)*AXT(I)
      BR(3,4,I,J,K)=A*(Q3(I)+C(I)*AYT(I))
      BR(3,5,I,J,K)=A*(Q3(I)-C(I)*AYT(I))
      BR(4,1,I,J,K)=Q4(I)*AXT(I)-Q1(I)*AYT(I)
      BR(4,2,I,J,K)=Q4(I)*AYT(I)+Q1(I)*AXT(I)
      BR(4,3,I,J,K)=Q4(I)*AZT(I)
      BR(4,4,I,J,K)=A*(Q4(I)+C(I)*AZT(I))
      BR(4,5,I,J,K)=A*(Q4(I)-C(I)*AZT(I))
      BR(5,1,I,J,K)=C1*AXT(I)+Q1(I)*(Q3(I)*AZT(I)-Q4(I)*AYT(I))
      BR(5,2,I,J,K)=C1*AYT(I)+Q1(I)*(Q4(I)*AXT(I)-Q2(I)*AZT(I))
      BR(5,3,I,J,K)=C1*AZT(I)+Q1(I)*(Q2(I)*AYT(I)-Q3(I)*AXT(I))
      BR(5,4,I,J,K)=A*(C1+C2+C(I)*TT(I))
      BR(5,5,I,J,K)=A*(C1+C2-C(I)*TT(I))
   23 CONTINUE
      DO 2 I=IB,gxefw
c      DO 2 I=IB,NI
      PHI=p5*GM1*RQRQ(I)
      C2=C(I)*C(I)
      B=RR2/RC(I)
      C1=one-PHI/C2
      C3=GM1/C2
      BL(1,1,I,J,K)=AXT(I)*C1+Q6(I)*(Q4(I)*AYT(I)-Q3(I)*AZT(I))
      BL(1,2,I,J,K)=AXT(I)*Q2(I)*C3
      BL(1,3,I,J,K)=AXT(I)*Q3(I)*C3+AZT(I)*Q6(I)
      BL(1,4,I,J,K)=AXT(I)*Q4(I)*C3-AYT(I)*Q6(I)
      BL(1,5,I,J,K)=-AXT(I)*C3
      BL(2,1,I,J,K)=AYT(I)*C1+Q6(I)*(Q2(I)*AZT(I)-Q4(I)*AXT(I))
      BL(2,2,I,J,K)=AYT(I)*Q2(I)*C3-AZT(I)*Q6(I)
      BL(2,3,I,J,K)=AYT(I)*Q3(I)*C3
      BL(2,4,I,J,K)=AYT(I)*Q4(I)*C3+AXT(I)*Q6(I)
      BL(2,5,I,J,K)=-AYT(I)*C3
      BL(3,1,I,J,K)=AZT(I)*C1+Q6(I)*(Q3(I)*AXT(I)-Q2(I)*AYT(I))
      BL(3,2,I,J,K)=AZT(I)*Q2(I)*C3+AYT(I)*Q6(I)
      BL(3,3,I,J,K)=AZT(I)*Q3(I)*C3-AXT(I)*Q6(I)
      BL(3,4,I,J,K)=AZT(I)*Q4(I)*C3
      BL(3,5,I,J,K)=-AZT(I)*C3
      BL(4,1,I,J,K)=B*(PHI-C(I)*TT(I))
      BL(4,2,I,J,K)=B*(C(I)*AXT(I)-Q2(I)*GM1)
      BL(4,3,I,J,K)=B*(C(I)*AYT(I)-Q3(I)*GM1)
      BL(4,4,I,J,K)=B*(C(I)*AZT(I)-Q4(I)*GM1)
      BL(4,5,I,J,K)=B*GM1
      BL(5,1,I,J,K)=B*(PHI+C(I)*TT(I))
      BL(5,2,I,J,K)=-B*(C(I)*AXT(I)+Q2(I)*GM1)
      BL(5,3,I,J,K)=-B*(C(I)*AYT(I)+Q3(I)*GM1)
      BL(5,4,I,J,K)=-B*(C(I)*AZT(I)+Q4(I)*GM1)
      BL(5,5,I,J,K)=B*GM1
    2 CONTINUE
      RETURN
      END

