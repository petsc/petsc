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
#include "es.h"
#include "sigma.h"
#include "fv.h"
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
      ierr = resid_fp(xx, p, dxx, br, bl, be,
     &         sadai, sadaj, sadak,
     &         aix, ajx, akx, aiy, ajy, aky, aiz, ajz, akz, 
     &         f1, g1, h1, sp, sm, sp1, sp2, sm1, sm2,
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
     &                  br,bl,be,sadai,sadaj,sadak,
     &                  aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                  f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2,
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
#include "es.h"
#include "sigma.h"
#include "condi.h"
#include "fv.h"
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
      double precision  DQ(5,D_NI),F(5,D_NI),qer(5),qel(5),tl
      integer i,j,k,l,ib,jb,kb,i1,j1,k1

      resid = 0

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
      xvel(i,j,k)=...
      yvel(i,j,k)=...
      zvel(i,j,k)=...
      den(i,j,k)=...
c
c   Form i-flux at the FV cell edge centers (I+1/2)
c
      F(6,I) = ( xvel(i,j,k)*aix(i,j,k) +
     *           yvel(i,j,k)*aiy(i,j,k) +
     *           zvel(i,j,k)*aiz(i,j,k) ) * den(i,j,k)
   21 continue
c
c   Form i-contribution to residual at cell centers (I,J,K) 
c
      DO 2 I=xsf2,xef01
      FP(I,J,K)=F(6,I)-F(6,I-1)
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
      xvel(i,j,k)=...
      yvel(i,j,k)=...
      zvel(i,j,k)=...
      den(i,j,k)=...
c
c   Form j-flux at the FV cell edge centers (J+1/2)
c
      F(6,J) = ( xvel(i,j,k)*ajx(i,j,k) +
     *           yvel(i,j,k)*ajy(i,j,k) +
     *           zvel(i,j,k)*ajz(i,j,k) ) * den(i,j,k)
   31 continue
c
c   Form j-contribution to residual at cell centers (I,J,K) 
c
      DO 3 J=ysf2,yef01
      FP(I,J,K)=F(6,J)-F(6,J-1)
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
      xvel(i,j,k)=...
      yvel(i,j,k)=...
      zvel(i,j,k)=...
      den(i,j,k)=...
c
c   Form k-flux at the FV cell edge centers (K+1/2)
c
      F(6,K) = ( xvel(i,j,k)*akx(i,j,k) +
     *           yvel(i,j,k)*aky(i,j,k) +
     *           zvel(i,j,k)*akz(i,j,k) ) * den(i,j,k)
   31 continue
c
c   Form k-contribution to residual at cell centers (I,J,K) 
c
      DO 4 K=zsf2,zef01
      FP(I,J,K)=F(6,K)-F(6,K-1)
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

      residbc = 0
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
c
c -------------------------------------------------------------------
c
      integer function jpressure(xx,p)
c
c jpress - Updates the pressure terms after a Newton step.
c
#include "param.h"
#include "varbls.h"
#include "consts.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
      double precision  rqdq
      integer i,j,k

      jpressure = 0
      do 15 k=gzsf1,gzefp1
      do 15 j=gysf1,gyefp1
      do 15 i=gxsf1,gxefp1
c      do 15 k=gzsf2,gzef01
c      do 15 j=gysf2,gyef01
c      do 15 i=gxsf2,gxef01
c      DO 15 K=2,NK
c      DO 15 J=2,NJ
c      DO 15 I=2,NI

c Update pressure through equation of state
      RQDQ=(RU(I,J,K)**2+RV(I,J,K)**2+
     1 RW(I,J,K)**2)/R(I,J,K)
      P(I,J,K)=GM1*(E(I,J,K)-p5*RQDQ)
   15 CONTINUE

      RETURN
      END

C ----------------------------------------------------------------

      SUBROUTINE RLVECS(QER,QEL,I1,J1,K1,IB,JB,KB,AX,AY,AZ,SADA,
     &                   xx,p,br,bl,be)
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
      double precision  SADA(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
c   Note: ax,ay,az must have same dimensions as arrays in avctrs.h
      double precision  ax(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
      double precision  ay(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)
      double precision  az(gxsf1:gxef01,gysf1:gyef01,gzsf1:gzef01)

      double precision  AXT(D_NI),AYT(D_NI),AZT(D_NI)
      double precision  Q1(D_NI),Q2(D_NI),Q3(D_NI),Q4(D_NI),Q6(D_NI),
     +                 RQRQ(D_NI),TT(D_NI),RC(D_NI),C(D_NI)
      double precision  qer(5),qel(5)
      double precision  a,b,c1,c2,c3,phi,rr2,e1,e4,e5,csad,hroe
      double precision  rl,rul,rvl,rwl,el,rr,rur,rvr,rwr,er,ra
      double precision  rra1,ur,vr,wr,ul,vl,wl,pr,pl,hr,hl
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

C ----------------------------------------------------------------

      SUBROUTINE VAlbada (dir,PSI,sp,sm,sp1,sp2,sm1,sm2)

c     VAlbada - The van Albada limiter. Computes the flux, SP().
c     See other limiters in the file xnd.F.

#include "param.h"
#include "sigma.h"
c      COMMON /SIGMA/ SP(5,NI),SM(5,NI)
c      COMMON /SIGMA/ SP1(5,NI),SP2(5,NI),SM1(5,NI),SM2(5,NI)
      double precision  eps,ydx,xdy,tp,tm,psi
      integer i,l,istart,iend,dir

      if (dir .eq. 1) then
        istart = gxsf2w
        iend   = xefm1
c        iend   = ni-1
      endif
      if (dir .eq. 2) then
        istart = gysf2w
        iend   = yefm1
c        iend   = nj-1
      endif
      if (dir .eq. 3) then
        istart = gzsf2w
        iend   = zefm1
c        iend   = nk-1
      endif

      EPS=1.d-12
      DO 10 L=1,5
      DO 10 I=istart,iend
      YDX=SM2(L,I)/(SM1(L,I) + EPS)
      TM=(YDX + YDX*YDX)/(1.0 + YDX*YDX) * SM1(L,I)
      XDY=SP2(L,I)/(SP1(L,I) + EPS)
      TP=(XDY + XDY*XDY)/(1.0 + XDY*XDY) * SP1(L,I)
      SP(L,I)=p5*(TP-TM)
   10 CONTINUE
      RETURN
      END

C ----------------------------------------------------------------

      SUBROUTINE DUMMY (dir,PSI,sp,sm,sp1,sp2,sm1,sm2)

c     DUMMY - No limiter.  Computes the flux, SP().

#include "param.h"
#include "sigma.h"
c      COMMON /SIGMA/ SP(5,NI),SM(5,NI)
c      COMMON /SIGMA/ SP1(5,NI),SP2(5,NI),SM1(5,NI),SM2(5,NI)
      double precision  psi
      integer i,l,istart,iend,dir

      if (dir .eq. 1) then
        istart = gxsf2w
        iend   = xefm1
c        iend   = ni-1
      endif
      if (dir .eq. 2) then
        istart = gysf2w
        iend   = yefm1
c        iend   = nj-1
      endif
      if (dir .eq. 3) then
        istart = gzsf2w
        iend   = zefm1
c        iend   = nk-1
      endif

      DO 10 I=istart,iend
      DO 10 L=1,5
      SP(L,I)=p5*(SP1(L,I)-SM1(L,I))
   10 CONTINUE
      RETURN
      END

