c
c  "$Id: resid_duct.F,v 1.5 1997/10/16 23:29:59 keyes Exp curfman $";
c
      integer function residbc_duct(jacform,xx,fff,p,xx_bc,p_bc,
     &                sadai,sadaj,sadak,
     &                aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)

c residbc - Forms the residual components due to boundary conditions.
c           This routine is called by localfortranfct().
c
c duct geometry definitions of ILE and ITU
c
c transpiration conditions for V over the "bump"
c
c                             ILE=17            ITU=33
c    |<---- domain ...        |<---- BUMP ----->|       ... domain ---->|
c    x1    x2    x3           x17               x33         x47   x48   x49
c o--|--o--|--o--| ...  |--o--|--o--| ... |--o--|--o--| ... |--o--|--o--|--o
c p1    p2    p3           p17   p18         p33   p34         p48   p49   p50
c
c Types of BCs:
c
c	for all (i,j): k = zsf1 and k=zef1 are impermeable
c
c	for all (j,k): i = xsf1 is subsonic inflow (U=const,V=W=0)
c		       i = xse1 is subsonic outflow
c
c	for all k:
c	     for i = xsf1 ... ILE:
c		       j = ysf1 is impermeable
c	     for i = ITU ... xse1:
c		       j = ysf1 is impermeable
c	     for i = ILE ... ITU:
c		       j = ysf1 is subsonic inflow (V=sin*cos,U=W=0)
c
c


#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "init.h"
#include "bc.h"
#include "ibcwork.h"

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
      double precision  ada,qda,ff,p0,r0,c0,pi1k,pb,u,v,w,rqdq,ri1k,p1jk
      double precision  u2,v2,w2,ub,con,r0c0,rb,vb,wb,pinj1k,pij1
      double precision  pijnk1,rni1jk,r1jk,ruimpo,rvimpo,rwimpo,xloc

      integer i,j,k,jacform

c     jacform - if 1, we are in the midst of forming the Jacobian via FD approx

      residbc_duct = 0
c
c subsonic inflow (transpiration)
c
      if (ysf1 .eq. 1) then
        do 2 k=zsf2,zef01
        do 2 i=xsf2,xef01
c      DO 2 K=2,NK
c      DO 2 I=2,NI
      IF (I.LE.ILE.OR.I.GT.ITU) GO TO 4
c
c Subsonic Inflow/Outflow for transpiration conditions at bump
c
C   THESE ARE CHARACTERISTIC VARIABLE SUBSONIC INFLOW B.C.s
      R0=p5*(R(I,2,K)+R(I,1,K))
      P0=p5*(P(I,2,K)+P(I,1,K))
      C0=SQRT(GAM*P0/R0)
      R0C0=R0*C0
      U2=RU(I,2,K)/R(I,2,K)
      V2=RV(I,2,K)/R(I,2,K)
      W2=RW(I,2,K)/R(I,2,K)
c
c Must use imposed transpiration, rather than INIT fields here
c
c      PB=p5*(PINIT+P(I,2,K)-R0C0/SADAJ(I,2,K)*(AJX(I,2,K)*(RUINIT-
c     & U2)+AJY(I,2,K)*(RVINIT-V2)+AJZ(I,2,K)*(RWINIT-W2)))
c      RB=RINIT+(PB-PINIT)/C0**2
c      CON=-(PINIT-PB)/(R0C0*SADAJ(I,2,K))
c      UB=RUINIT+AJX(I,2,K)*CON
c      VB=RVINIT+AJY(I,2,K)*CON
c      WB=RWINIT+AJZ(I,2,K)*CON
      RUIMPO=0.0D0
      xloc=-1.50d0+3.0d0*(dfloat(i-1)-p5)/dfloat(ni-1)
      RVIMPO=bump*fsmach*pi*sin(pi*xloc)*cos(pi*xloc)
      RWIMPO=0.0D0
      PB=p5*(PINIT+P(I,2,K)-R0C0/SADAJ(I,2,K)*(AJX(I,2,K)*(RUIMPO-
     & U2)+AJY(I,2,K)*(RVIMPO-V2)+AJZ(I,2,K)*(RWIMPO-W2)))
      RB=RINIT+(PB-PINIT)/C0**2
      CON=-(PINIT-PB)/(R0C0*SADAJ(I,2,K))
      UB=RUIMPO+AJX(I,2,K)*CON
      VB=RVIMPO+AJY(I,2,K)*CON
      WB=RWIMPO+AJZ(I,2,K)*CON
      PI1K=2.*PB-P(I,2,K)
      RI1K=2.*RB-R(I,2,K)
      fbcrj1(I,K)=R(I,1,K)-2.*RB+R(I,2,K)
      U=2.*UB-U2
      V=2.*VB-V2
      W=2.*WB-W2
      fbcruj1(i,k)=RU(I,1,K)-U*RI1K
      fbcrvj1(i,k)=RV(I,1,K)-V*RI1K
      fbcrwj1(i,k)=RW(I,1,K)-W*RI1K
      fbcej1(i,k)=E(I,1,K)-PI1K*RGM1
     &                        -p5*RI1K*(U**2+V**2+W**2)
      GOTO 2 
c
    4 CONTINUE
c
c Impermeable wall at bottom of duct away from "bump"
c
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
    2 CONTINUE
      endif
c
c Impermeable wall at top of duct
c
      if (yef01 .eq. nj) then
        do 20 k=zsf2,zef01
        do 20 i=xsf2,xef01
c      DO 20 K=2,NK
c      DO 20 I=2,NI
C   THESE ARE CHARACTERISTIC VARIABLE IMPERMEABLE WALL B.C.s

      QDA=(RU(I,NJ,K)*AJX(I,NJ,K)+RV(I,NJ,K)*AJY(I,NJ,K)
     *   +RW(I,NJ,K)*AJZ(I,NJ,K))/R(I,NJ,K)
      P0=p5*(P(I,NJ1,K)+P(I,NJ,K))
      R0=p5*(R(I,NJ1,K)+R(I,NJ,K))
      C0=SQRT(GAM*P0/R0)
      PB=P(I,NJ,K)-R0*C0*QDA/SADAJ(I,NJ,K)
      PINJ1K=2.*PB-P(I,NJ,K)
      fbcrj2(i,k)=
     &  r(I,NJ1,K)-R(I,NJ,K)-2.*(PB-P(I,NJ,K))/C0**2
      FF=two*QDA/SADAJ(I,NJ,K)**2
      U=RU(I,NJ,K)/R(I,NJ,K)-FF*AJX(I,NJ,K)
      V=RV(I,NJ,K)/R(I,NJ,K)-FF*AJY(I,NJ,K)
      W=RW(I,NJ,K)/R(I,NJ,K)-FF*AJZ(I,NJ,K)
      fbcruj2(i,k)=ru(I,NJ1,K)-U*R(I,NJ1,K)
      fbcrvj2(i,k)=rv(I,NJ1,K)-V*R(I,NJ1,K)
      fbcrwj2(i,k)=rw(I,NJ1,K)-W*R(I,NJ1,K)
      RQDQ=p5*(RU(I,NJ1,K)**2+RV(I,NJ1,K)**2+RW(I,NJ1,K)**2)/
     &   R(I,NJ1,K)
      fbcej2(i,k)=
     &  e(i,NJ1,k)-PINJ1K*RGM1-RQDQ
   20 CONTINUE
      endif
c
c   Symmetry Walls on inboard and outboard walls of duct
c
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
      if (zef01 .eq. nk_boundary) then
        do 7 j=ysf2,yef01
        do 7 i=xsf2,xef01
      ADA=AKX(I,J,NK_BOUNDARY)**2+AKY(I,J,NK_BOUNDARY)**2
     *   +AKZ(I,J,NK_BOUNDARY)**2
      QDA=RU(I,J,NK_BOUNDARY)*AKX(I,J,NK_BOUNDARY)
     *   +RV(I,J,NK_BOUNDARY)*AKY(I,J,NK_BOUNDARY)
     *   +RW(I,J,NK_BOUNDARY)*AKZ(I,J,NK_BOUNDARY)
      fbcrk2(i,j)=R(I,J,NK1_BOUNDARY)
     *   -R(I,J,NK_BOUNDARY)
      fbcek2(i,j)=E(I,J,NK1_BOUNDARY)
     *   -E(I,J,NK_BOUNDARY)
      PIJNK1=P(I,J,NK_BOUNDARY)
      FF=two*QDA/ADA
      fbcruk2(i,j)=RU(I,J,NK1_BOUNDARY)-
     *  RU(I,J,NK_BOUNDARY)+FF*AKX(I,J,NK_BOUNDARY)
      fbcrvk2(i,j)=RV(I,J,NK1_BOUNDARY)-
     *  RV(I,J,NK_BOUNDARY)+FF*AKY(I,J,NK_BOUNDARY)
      fbcrwk2(i,j)=RW(I,J,NK1_BOUNDARY)-
     *  RW(I,J,NK_BOUNDARY)+FF*AKZ(I,J,NK_BOUNDARY)
    7 CONTINUE
      endif
c
c Subsonic Outflow at duct exit
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
     &                        -p5*RNI1JK*(U**2+V**2+W**2)
    8 CONTINUE
      endif
c
c Subsonic Inflow at duct entrance
c
      if (xsf1 .eq. 1) then
        do 9 k=zsf2,zef01
        do 9 j=ysf2,yef01
C   THESE ARE CHARACTERISTIC VARIABLE SUBSONIC INFLOW B.C.s
      R0=p5*(R(1,J,K)+R(2,J,K))
      P0=p5*(P(1,J,K)+P(2,J,K))
      C0=SQRT(GAM*P0/R0)
      R0C0=R0*C0
      U2=RU(2,J,K)/R(2,J,K)
      V2=RV(2,J,K)/R(2,J,K)
      W2=RW(2,J,K)/R(2,J,K)
      PB=p5*(PINIT+P(2,J,K)-R0C0/SADAI(2,J,K)*(AIX(2,J,K)*(RUINIT-
     & U2)+AIY(2,J,K)*(RVINIT-V2)+AIZ(2,J,K)*(RWINIT-W2)))
      RB=RINIT+(PB-PINIT)/C0**2
      CON=-(PINIT-PB)/(R0C0*SADAI(2,J,K))
      UB=RUINIT+AIX(2,J,K)*CON
      VB=RVINIT+AIY(2,J,K)*CON
      WB=RWINIT+AIZ(2,J,K)*CON
      P1JK=2.*PB-P(2,J,K)
      R1JK=2.*RB-R(2,J,K)
      fbcri1(j,k)=R(1,J,K)-2.*RB+R(2,J,K)
      U=2.*UB-U2
      V=2.*VB-V2
      W=2.*WB-W2
      fbcrui1(j,k)=RU(1,J,K)-U*R1JK
      fbcrvi1(j,k)=RV(1,J,K)-V*R1JK
      fbcrwi1(j,k)=RW(1,J,K)-W*R1JK
      fbcei1(j,k)=E(1,J,K)-P1JK*RGM1
     &    -p5*R1JK*(U**2+V**2+W**2)
    9 CONTINUE
      endif

      RETURN
      END
c

