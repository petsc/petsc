c
c  "$Id: ex1f.F,v 1.16 1997/09/22 15:20:58 balay Exp $";
c
c  This file contains routines to evaluate the local Fortran function.
c  When using automatic differentiation tools, all of these routines
c  require processing.
c
c -------------------------------------------------------------------
c
      integer function localfortfct_fp(first_time_resid,fff,
     &                xx,p,xx_bc,p_bc,sadai,sadaj,sadak,
     &                aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                dxx,den,xvel,yvel,zvel)

c localfortfct - Evaluates local Fortran function (using implicit BCs).
c                We use a single routine to interface between the C routine
c                ComputeFunction() and the various Fortran work routines 
c                to facilitate use of automatic differentiation.

#include "param.h"
#include "varbls.h"
#include "dvarbls.h"
#include "avctrs.h"
#include "sada.h"
#include "bc.h"
#include "ibcwork.h"
#include "aux_fp.h"

      integer first_time_resid,ierr,resid_fp,residbc_fp,jpressure
  
      localfortfct_fp = 0

C  Compute pressures 
      ierr = jpressure(xx, p)
      if (ierr .ne. 0) then
         write(6,*)  'localfortfct:rank,jpressure',
     &                rank,ierr
         localfortfct_fp = -1
         return
      endif

C  If using implicit BC's, compute function components for boundary points 
      ierr = residbc_fp(xx, fff, p, xx_bc, p_bc, sadai, sadaj, sadak,
     &         aix, ajx, akx, aiy, ajy, aky, aiz, ajz, akz,
     &         den, xvel, yvel, zvel)
      if (ierr .ne. 0) then
         write(6,*)  'local_fortran_fct:rank,residbc',
     &                rank,ierr
         localfortfct_fp = -1
         return
      endif

C  Compute function components for interior grid points
      ierr = resid_fp(xx, p, dxx,
     &         sadai, sadaj, sadak,
     &         aix, ajx, akx, aiy, ajy, aky, aiz, ajz, akz, 
     &         den, xvel, yvel, zvel)
      if (ierr .ne. 0) then
         write(6,*)  'local_fortran_fct:rank,resid',
     &                rank,ierr
         localfortfct_fp = -1
         return
      endif

      return
      end

C ----------------------------------------------------------------

      integer function RESID_fp(xx,p,dxx,
     &                  sadai,sadaj,sadak,
     &                  aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                  den,xvel,yvel,zvel)

c Computes the nonlinear residual (five components at each interior point)
c Each coordinate direction is considered in turn (two cell faces per
c coordinate direction), amortizing use of one-dimensional logic in RLVECS,
c though most of this is written out more explicitly than necessary.

#include "param.h"
#include "varbls.h"
#include "dvarbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "freq.h"
#include "condi.h"
#include "aux_fp.h"

C   RESID COMPUTES THE RESIDUAL VECTOR.
c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c      COMMON /DVRBLS/ DR(NI1,NJ1,NK1),DRU(NI1,NJ1,NK1),DRV(NI1,NJ1,NK1)
c      COMMON /DVRBLS/ DRW(NI1,NJ1,NK1),DE(NI1,NJ1,NK1)
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
c      COMMON /ES/ BR(5,5,NI,NJ,NK),BL(5,5,NI,NJ,NK),BE(2,5,NI,NJ,NK)
c      common /fv/ f1(5,ni,nj,nk),g1(5,ni,nj,nk),h1(5,ni,nj,nk)
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /SIGMA/ SP(5,NI),SM(5,NI)
c      COMMON /SIGMA/ SP1(5,NI),SP2(5,NI),SM1(5,NI),SM2(5,NI)
c      COMMON /CONDI/ ORDER,PSI,LIMIT
c      COMMON /FREQ/ IFREQ,ISGS
      double precision F(D_NI)
      integer i,j,k

      resid_fp = 0

C   FLUX IN THE I DIRECTION
      do 2 k=zsf2,zef01
      do 2 j=ysf2,yef01
      do 21 i=gxsf1,gxefw
c      DO 2 K=2,NK
c      DO 2 J=2,NJ
c      DO 21 I=1,NI
c
c   Form density and contravariant velocities at FV cell edge centers (I+1/2)
c   as functions of PHI(I,J,K) and PHI(I+1,J,K)
c
c      xvel(i,j,k)=...
c      yvel(i,j,k)=...
c      zvel(i,j,k)=...
c      den(i,j,k)=...
c
c   Form i-flux at the FV cell edge centers (I+1/2)
c
      F(I) = ( xvel(i,j,k)*aix(i,j,k) +
     *           yvel(i,j,k)*aiy(i,j,k) +
     *           zvel(i,j,k)*aiz(i,j,k) ) * den(i,j,k)
   21 continue
c
c   Form i-contribution to residual at cell centers (I,J,K) 
c
      DO 2 I=xsf2,xef01
      FP(I,J,K)=F(I)-F(I-1)
    2 CONTINUE

C   FLUX IN THE J DIRECTION
      do 3 k=zsf2,zef01
      do 3 i=xsf2,xef01
      do 31 j=gysf1,gyefw
c      DO 3 K=2,NK
c      DO 3 I=2,NI
c      DO 31 J=1,NJ
c
c   Form density and contravariant velocities at FV cell edge centers (J+1/2)
c   as functions of PHI(I,J,K) and PHI(I,J+1,K)
c
c      xvel(i,j,k)=...
c      yvel(i,j,k)=...
c      zvel(i,j,k)=...
c      den(i,j,k)=...
c
c   Form j-flux at the FV cell edge centers (J+1/2)
c
      F(J) = ( xvel(i,j,k)*ajx(i,j,k) +
     *           yvel(i,j,k)*ajy(i,j,k) +
     *           zvel(i,j,k)*ajz(i,j,k) ) * den(i,j,k)
   31 continue
c
c   Form j-contribution to residual at cell centers (I,J,K) 
c
      DO 3 J=ysf2,yef01
      FP(I,J,K) = FP(I,J,K) + F(J)-F(J-1)
    3 CONTINUE

C   THE FOLLOWING IS FOR 2-D PLANAR FLOW
C     RETURN
C   FLUX IN THE K DIRECTION
      do 4 j=ysf2,yef01
      do 4 i=xsf2,xef01
      do 41 k=gzsf1,gzefw
c      DO 4 J=2,NJ
c      DO 4 I=2,NI
c      DO 41 K=1,NK
c
c   Form density and contravariant velocities at FV cell edge centers (K+1/2)
c   as functions of PHI(I,J,K) and PHI(I,J,K+1)
c
c      xvel(i,j,k)=...
c      yvel(i,j,k)=...
c      zvel(i,j,k)=...
c      den(i,j,k)=...
c
c   Form k-flux at the FV cell edge centers (K+1/2)
c
      F(K) = ( xvel(i,j,k)*akx(i,j,k) +
     *           yvel(i,j,k)*aky(i,j,k) +
     *           zvel(i,j,k)*akz(i,j,k) ) * den(i,j,k)
   41 continue
c
c   Form k-contribution to residual at cell centers (I,J,K) 
c
      DO 4 K=zsf2,zef01
      FP(I,J,K) = FP(I,J,K) + F(K)-F(K-1)
    4 CONTINUE

      RETURN
      END

c ---------------------------------------------------------------

      integer function residbc_fp(xx,fff,p,xx_bc,p_bc,
     &                sadai,sadaj,sadak,
     &                aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                den,xvel,yvel,zvel)

c residbc - Forms the residual components due to boundary conditions.
c           This routine is called by localfortranfct().
c
c           This routine used as a starting point similar uniprocessor
c           code written by M.D. Tidriri.

#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "init.h"
#include "bc.h"
#include "ibcwork.h"
#include "aux_fp.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /COORD/ X(NI,NJ,NK),Y(NI,NJ,NK),Z(NI,NJ,NK)
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /INIT/ RINIT,RUINIT,RVINIT,RWINIT,EINIT
      double precision  ada,qda,ff,p0,r0,c0,pi1k,pb,u,v,w,rqdq
      double precision  u2,v2,w2,ub,con,r0c0,rb,vb,wb,pinj1k,rinj1k,pij1
      double precision  pijnk1,rni1jk,r1jk

      integer i,j,k

      residbc_fp = 0
c
      if (ysf1 .eq. 1) then
        do 2 k=zsf2,zef01
        do 2 i=xsf2,xef01
c      DO 2 K=2,NK
c      DO 2 I=2,NI
      IF (K.GT.KTIP) GO TO 4
      IF (I.LE.ITL.OR.I.GT.ITU) GO TO 4
      IF (NCYC.GE.bcswitch) GO TO 15
      ADA=SADAJ(I,1,K)**2
      QDA=RU(I,2,K)*AJX(I,1,K)+RV(I,2,K)*AJY(I,1,K)
     *   +RW(I,2,K)*AJZ(I,1,K)
      FF=two*QDA/ADA
      fbcruj1(i,k)=ru(I,1,K)-RU(I,2,K)+FF*AJX(I,1,K)
      fbcrvj1(i,k)=rv(I,1,K)-RV(I,2,K)+FF*AJY(I,1,K)
      fbcrwj1(i,k)=rw(I,1,K)-RW(I,2,K)+FF*AJZ(I,1,K)
      fbcrj1(i,k)=r(I,1,K)-R(I,2,K)
      fbcej1(i,k)=e(I,1,K)-E(I,2,K)
      GO TO 2
   15 CONTINUE
C   THESE ARE CHARACTERISTIC VARIABLE IMPERMEABLE WALL B.C.s
      QDA=(RU(I,2,K)*AJX(I,1,K)+RV(I,2,K)*AJY(I,1,K)
     *   +RW(I,2,K)*AJZ(I,1,K))/R(I,2,K)
      P0=p5*(P(I,1,K)+P(I,2,K))
      R0=p5*(R(I,1,K)+R(I,2,K))
      C0=SQRT(GAM*P0/R0)
      PB=P(I,2,K)-R0*C0*QDA/SADAJ(I,1,K)
      PI1K=2.*PB-P(I,2,K)
      fbcrj1(i,k)=
     &  r(I,1,K)-R(I,2,K)-2.*(PB-P(I,2,K))/C0**2
      FF=two*QDA/SADAJ(I,1,K)**2
      U=RU(I,2,K)/R(I,2,K)-FF*AJX(I,1,K)
      V=RV(I,2,K)/R(I,2,K)-FF*AJY(I,1,K)
      W=RW(I,2,K)/R(I,2,K)-FF*AJZ(I,1,K)
      fbcruj1(i,k)=ru(I,1,K)-U*R(I,1,K)
      fbcrvj1(i,k)=rv(I,1,K)-V*R(I,1,K)
      fbcrwj1(i,k)=rw(I,1,K)-W*R(I,1,K)
      RQDQ=p5*(RU(I,1,K)**2+RV(I,1,K)**2+RW(I,1,K)**2)/R(I,1,K)
      fbcej1(i,k)=
     &  e(i,1,k)-PI1K*RGM1-RQDQ
      GO TO 2
    4 CONTINUE
cc      ID=NI+2-I
c      ID=NI+1-I
      fbcruj1(i,k)=ru(I,1,K)-RU_bc(I,2,K)
      fbcrvj1(i,k)=rv(I,1,K)-RV_bc(I,2,K)
      fbcrwj1(i,k)=rw(I,1,K)-RW_bc(I,2,K)
      fbcrj1(i,k)=r(I,1,K)-R_bc(I,2,K)
      fbcej1(i,k)=e(I,1,K)-E_bc(I,2,K)
c      fbcruj1(i,k)=ru(I,1,K)-RU(ID,2,K)
c      fbcrvj1(i,k)=rv(I,1,K)-RV(ID,2,K)
c      fbcrwj1(i,k)=rw(I,1,K)-RW(ID,2,K)
c      fbcrj1(i,k)=r(I,1,K)-R(ID,2,K)
c      fbcej1(i,k)=e(I,1,K)-E(ID,2,K)
    2 CONTINUE
      endif
c
      if (yef01 .eq. nj) then
        do 20 k=zsf2,zef01
        do 20 i=xsf2,xef01
c      DO 20 K=2,NK
c      DO 20 I=2,NI
C   THESE ARE CHARACTERISTIC VARIABLE SUBSONIC INFLOW B.C.s
      R0=p5*(R(I,NJ,K)+R(I,NJ1,K))
      P0=p5*(P(I,NJ,K)+P(I,NJ1,K))
      C0=SQRT(GAM*P0/R0)
      R0C0=R0*C0
      U2=RU(I,NJ,K)/R(I,NJ,K)
      V2=RV(I,NJ,K)/R(I,NJ,K)
      W2=RW(I,NJ,K)/R(I,NJ,K)
      PB=p5*(PINIT+P(I,NJ,K)-R0C0/SADAJ(I,NJ,K)*(AJX(I,NJ,K)*(RUINIT-
     * U2)+AJY(I,NJ,K)*(RVINIT-V2)+AJZ(I,NJ,K)*(RWINIT-W2)))
      RB=RINIT+(PB-PINIT)/C0**2
      CON=-(PINIT-PB)/(R0C0*SADAJ(I,NJ,K))
      UB=RUINIT+AJX(I,NJ,K)*CON
      VB=RVINIT+AJY(I,NJ,K)*CON
      WB=RWINIT+AJZ(I,NJ,K)*CON
      PINJ1K=2.*PB-P(I,NJ,K)
      RINJ1K=2.*RB-R(I,NJ,K)
      fbcrj2(I,K)=R(I,NJ1,K)-2.*RB+R(I,NJ,K)
      U=2.*UB-U2
      V=2.*VB-V2
      W=2.*WB-W2
      fbcruj2(i,k)=RU(I,NJ1,K)-U*RINJ1K
      fbcrvj2(i,k)=RV(I,NJ1,K)-V*RINJ1K
      fbcrwj2(i,k)=RW(I,NJ1,K)-W*RINJ1K
      fbcej2(i,k)=E(I,NJ1,K)-PINJ1K*RGM1
     s                        -p5*RINJ1K*(U**2+V**2+W**2)
   20 CONTINUE
      endif

      if (zsf1 .eq. 1) then
        do 6 j=ysf2,yef01
        do 6 i=xsf2,xef01
c      DO 6 J=2,NJ
c      DO 6 I=2,NI
      ADA=AKX(I,J,1)**2+AKY(I,J,1)**2+AKZ(I,J,1)**2
      QDA=RU(I,J,2)*AKX(I,J,1)+RV(I,J,2)*AKY(I,J,1)
     *   +RW(I,J,2)*AKZ(I,J,1)
      fbcrk1(i,j)=R(I,J,1)-R(I,J,2)
      fbcek1(i,j)=E(I,J,1)-E(I,J,2)
      PIJ1=P(I,J,2)
      FF=two*QDA/ADA
      fbcruk1(i,j)=RU(I,J,1)-RU(I,J,2)+FF*AKX(I,J,1)
      fbcrvk1(i,j)=RV(I,J,1)-RV(I,J,2)+FF*AKY(I,J,1)
      fbcrwk1(i,j)=RW(I,J,1)-RW(I,J,2)+FF*AKZ(I,J,1)
    6 CONTINUE
      endif
c
      if (zef01 .eq. nk) then
        do 7 j=ysf2,yef01
        do 7 i=xsf2,xef01
      ADA=AKX(I,J,NK)**2+AKY(I,J,NK)**2+AKZ(I,J,NK)**2
      QDA=RU(I,J,NK)*AKX(I,J,NK)+RV(I,J,NK)*AKY(I,J,NK)
     *   +RW(I,J,NK)*AKZ(I,J,NK)
      fbcrk2(i,j)=R(I,J,NK1)-R(I,J,NK)
      fbcek2(i,j)=E(I,J,NK1)-E(I,J,NK)
      PIJNK1=P(I,J,NK)
      FF=two*QDA/ADA
      fbcruk2(i,j)=RU(I,J,NK1)-RU(I,J,NK)+FF*AKX(I,J,NK)
      fbcrvk2(i,j)=RV(I,J,NK1)-RV(I,J,NK)+FF*AKY(I,J,NK)
      fbcrwk2(i,j)=RW(I,J,NK1)-RW(I,J,NK)+FF*AKZ(I,J,NK)
    7 CONTINUE
      endif
c
      if (xef01 .eq. ni) then
        do 8 k=zsf2,zef01
        do 8 j=ysf2,yef01
c      DO 8 K=2,NK
c      DO 8 J=2,NJ
C   THESE ARE CHARACTERISTIC VARIABLE SUBSONIC OUTFLOW B.C.s
      R0=p5*(R(NI1,J,K)+R(NI,J,K))
      C0=SQRT(GAM*PINIT/R0)
      RNI1JK=R(NI,J,K)+2.*(PINIT-P(NI,J,K))/C0**2
      fbcri2(j,k)=
     &  R(NI1,J,K)-R(NI,J,K)-2.*(PINIT-P(NI,J,K))/C0**2
      CON=-two*(PINIT-P(NI,J,K))/(SADAI(NI,J,K)*R0*C0)
      U=RU(NI,J,K)/R(NI,J,K)+AIX(NI,J,K)*CON
      V=RV(NI,J,K)/R(NI,J,K)+AIY(NI,J,K)*CON
      W=RW(NI,J,K)/R(NI,J,K)+AIZ(NI,J,K)*CON
      fbcrui2(j,k)=RU(NI1,J,K)-U*RNI1JK
      fbcrvi2(j,k)=RV(NI1,J,K)-V*RNI1JK
      fbcrwi2(j,k)=RW(NI1,J,K)-W*RNI1JK
      fbcei2(j,k)=E(NI1,J,K)-PINIT*RGM1
     s                        -p5*RNI1JK*(U**2+V**2+W**2)
    8 CONTINUE
      endif
c
      if (xsf1 .eq. 1) then
        do 9 k=zsf2,zef01
        do 9 j=ysf2,yef01
      R0=p5*(R(1,J,K)+R(2,J,K))
      C0=SQRT(GAM*PINIT/R0)
      R1JK=R(2,J,K)+2.*(PINIT-P(2,J,K))/C0**2
      fbcri1(j,k)=R(1,J,K)-R(2,J,K)-2.*(PINIT-P(2,J,K))/C0**2
      CON=two*(PINIT-P(2,J,K))/(SADAI(1,J,K)*R0*C0)
      U=RU(2,J,K)/R(2,J,K)+AIX(1,J,K)*CON
      V=RV(2,J,K)/R(2,J,K)+AIY(1,J,K)*CON
      W=RW(2,J,K)/R(2,J,K)+AIZ(1,J,K)*CON
      fbcrui1(j,k)=RU(1,J,K)-U*R1JK
      fbcrvi1(j,k)=RV(1,J,K)-V*R1JK
      fbcrwi1(j,k)=RW(1,J,K)-W*R1JK
      fbcei1(j,k)=E(1,J,K)-PINIT*RGM1-p5*R1JK*(U**2+V**2+W**2)
    9 CONTINUE
      endif

      RETURN
      END


