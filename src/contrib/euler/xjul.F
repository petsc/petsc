C
C  This file contains routines that are used only by the Julianne solver,
C  not by any of the PETSc solvers.
C

C************************************************************************

      SUBROUTINE BC_UNI(xx,p,sadai,sadaj,sadak,
     &                  aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
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
      Double  ada,qda,ff,p0,r0,c0,pb,u,v,w,rqdq,r0c0,ub
      Double  u2,v2,w2,con,vb,wb,rb
      integer i,j,k,id

      if (size .gt. 1) then
        if (rank .eq. 0) write(6,*) 'bc_uni: uniprocessor only!'
        stop
      endif
      bcswitch = 10

C  Note that we MUST set i=1 boundary first, since the fields at points
C  (i=1,j=2,k=<anything>) are used to set the (i=49,j=1,k=<anything>) boundary.
      DO 8 K=2,NK
      DO 8 J=2,NJ
C   THESE ARE CHARACTERISTIC VARIABLE SUBSONIC OUTFLOW B.C.s
      R0=p5*(R(NI1,J,K)+R(NI,J,K))
      C0=SQRT(GAM*PINIT/R0)
      R(NI1,J,K)=R(NI,J,K)+2.*(PINIT-P(NI,J,K))/C0**2
      CON=-two*(PINIT-P(NI,J,K))/(SADAI(NI,J,K)*R0*C0)
      U=RU(NI,J,K)/R(NI,J,K)+AIX(NI,J,K)*CON
      V=RV(NI,J,K)/R(NI,J,K)+AIY(NI,J,K)*CON
      W=RW(NI,J,K)/R(NI,J,K)+AIZ(NI,J,K)*CON
      RU(NI1,J,K)=U*R(NI1,J,K)
      RV(NI1,J,K)=V*R(NI1,J,K)
      RW(NI1,J,K)=W*R(NI1,J,K)
      E(NI1,J,K)=PINIT*RGM1+p5*R(NI1,J,K)*(U**2+V**2+W**2)
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
    8 CONTINUE

      DO 2 K=2,NK
      DO 2 I=2,NI
      IF (K.GT.KTIP) GO TO 4
      IF (I.LE.ITL.OR.I.GT.ITU) GO TO 4
      IF (NCYC.GT.bcswitch) GO TO 15
      ADA=SADAJ(I,1,K)**2
      QDA=RU(I,2,K)*AJX(I,1,K)+RV(I,2,K)*AJY(I,1,K)
     *   +RW(I,2,K)*AJZ(I,1,K)
      FF=two*QDA/ADA
      RU(I,1,K)=RU(I,2,K)-FF*AJX(I,1,K)
      RV(I,1,K)=RV(I,2,K)-FF*AJY(I,1,K)
      RW(I,1,K)=RW(I,2,K)-FF*AJZ(I,1,K)
      P(I,1,K)=P(I,2,K)
      R(I,1,K)=R(I,2,K)
      E(I,1,K)=E(I,2,K)
      GO TO 2
   15 CONTINUE
C   THESE ARE CHARACTERISTIC VARIABLE IMPERMEABLE WALL B.C.s
      QDA=(RU(I,2,K)*AJX(I,1,K)+RV(I,2,K)*AJY(I,1,K)
     *   +RW(I,2,K)*AJZ(I,1,K))/R(I,2,K)
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
    4 CONTINUE
      ID=NI+1-I
c      ID=NI+2-I
      R(I,1,K)=R(ID,2,K)
      P(I,1,K)=P(ID,2,K)
      E(I,1,K)=E(ID,2,K)
      RU(I,1,K)=RU(ID,2,K)
      RV(I,1,K)=RV(ID,2,K)
      RW(I,1,K)=RW(ID,2,K)
    2 CONTINUE

      DO 20 K=2,NK
      DO 20 I=2,NI
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
      P(I,NJ1,K)=2.*PB-P(I,NJ,K)
      R(I,NJ1,K)=2.*RB-R(I,NJ,K)
      U=2.*UB-U2
      V=2.*VB-V2
      W=2.*WB-W2
      RU(I,NJ1,K)=U*R(I,NJ1,K)
      RV(I,NJ1,K)=V*R(I,NJ1,K)
      RW(I,NJ1,K)=W*R(I,NJ1,K)
      E(I,NJ1,K)=P(I,NJ1,K)*RGM1+p5*R(I,NJ1,K)*(U**2+V**2+W**2)
   20 CONTINUE
      DO 6 J=2,NJ
      DO 6 I=2,NI
      ADA=AKX(I,J,1)**2+AKY(I,J,1)**2+AKZ(I,J,1)**2
      QDA=RU(I,J,2)*AKX(I,J,1)+RV(I,J,2)*AKY(I,J,1)
     *   +RW(I,J,2)*AKZ(I,J,1)
      R(I,J,1)=R(I,J,2)
      E(I,J,1)=E(I,J,2)
      P(I,J,1)=P(I,J,2)
      FF=two*QDA/ADA
      RU(I,J,1)=RU(I,J,2)-FF*AKX(I,J,1)
      RV(I,J,1)=RV(I,J,2)-FF*AKY(I,J,1)
      RW(I,J,1)=RW(I,J,2)-FF*AKZ(I,J,1)
      ADA=AKX(I,J,NK)**2+AKY(I,J,NK)**2+AKZ(I,J,NK)**2
      QDA=RU(I,J,NK)*AKX(I,J,NK)+RV(I,J,NK)*AKY(I,J,NK)
     *   +RW(I,J,NK)*AKZ(I,J,NK)
      R(I,J,NK1)=R(I,J,NK)
      E(I,J,NK1)=E(I,J,NK)
      P(I,J,NK1)=P(I,J,NK)
      FF=two*QDA/ADA
      RU(I,J,NK1)=RU(I,J,NK)-FF*AKX(I,J,NK)
      RV(I,J,NK1)=RV(I,J,NK)-FF*AKY(I,J,NK)
      RW(I,J,NK1)=RW(I,J,NK)-FF*AKZ(I,J,NK)
    6 CONTINUE

      RETURN
      END

C************************************************************************

      SUBROUTINE STEP(time_init,eps,epsi,rtol,converge,
     &                 b1,b2,b3,b4,b5,b6,d,dt,
     &                 xx,p,dxx,
     &                 br,bl,be,sadai,sadaj,sadak,
     &                 aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                 f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)

c STEP is the main driver routine, executing a solve at each time step
c Uniprocessor version only!
c
#include "param.h"
#include "matrixb.h"
#include "diag.h"
#include "dt.h"
#include "varbls.h"
#include "dvarbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "freq.h"
#include "ang.h"
#include "gs.h"
#include "es.h"
#include "fv.h"
#include "sigma.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
c      common /gs/ res(5,ni,nj,nk),vec(5,ni,nj,nk)
c      COMMON /TSTEP/ DT(NI,NJ,NK)
c      COMMON /ANG/ ALPHA,BETA,PHI
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /FREQ/ IFREQ,ISGS

      Double  qer(5),qel(5),eps,dql,dqr,xl2,rqdq,rtmax,rtrms
      Double  etmax,etrms,con,rt,et,rtrmsr,etrmsr,tcl,tcd
      Double  areas,cn,cc,area,pbody,cls,cds,xl2a,epsi
      Double  rtol,xl2_init,time_init
      integer i,j,k,l,m,nsup,converge

      do 11 m=1,5
         qel(m)=zero
         qer(m)=zero
 11   continue

      NCYC=NCYC+1
c Evaluate nonlinear residual

      CALL RESID(xx,p,dxx,
     &           br,bl,be,sadai,sadaj,sadak,
     &           aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &           f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
 
      IF (NCYC.NE.1.AND.MOD(NCYC,IFREQ).NE.0) GO TO 10
      do 30 m=1,5
      qel(m)=eps
c Form Jacobian elements in left part of matrix
      call nd (m,epsi,qer,qel,b1,b2,b3,
     &         xx,p,
     &         br,bl,be,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &         f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
      qel(m)=0.
   30 continue
      do 31 m=1,5
      qer(m)=eps
c Form Jacobian elements in right part of matrix
      call nd (m,epsi,qer,qel,b4,b5,b6,
     &         xx,p,
     &         br,bl,be,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &         f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
      qer(m)=0.
   31 continue

c Find eigendecomposition of diagonal block of Jacobian
      call eigenv(dt,xx,p,sadai,sadaj,sadak,
     &            aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
      DO 21 M=1,5
      DO 21 L=1,5
      DO 21 K=2,NK
      DO 21 J=2,NJ
      DO 21 I=2,NI
      dql=B1(L,M,I,J,K)+B2(L,M,I,J,K)+B3(L,M,I,J,K)
      dqr=B4(L,M,I-1,J,K)+B5(L,M,I,J-1,K)+B6(L,M,I,J,K-1)
      D(L,M,I,J,K)=dql-dqr
      IF (L.EQ.M) D(L,M,I,J,K)=D(L,M,I,J,K)+one/DT(I,J,K)
   21 CONTINUE

c Decompose diagonal block into LU factors
      CALL DEQLU(d)
   10 CONTINUE
c Compute Euclidean norm of residual
      XL2A=0.
      DO 12 K=2,NK
      DO 12 J=2,NJ
      DO 12 I=2,NI
      XL2A=XL2A+DR(I,J,K)**2+DRU(I,J,K)**2+DRV(I,J,K)**2+
     *        DRW(I,J,K)**2+DE(I,J,K)**2
c Set up negative of nonlinear residual as the right-hand side
      res(1,i,j,k)=-dr(i,j,k)
      res(2,i,j,k)=-dru(i,j,k)
      res(3,i,j,k)=-drv(i,j,k)
      res(4,i,j,k)=-drw(i,j,k)
      res(5,i,j,k)=-de(i,j,k)
   12 CONTINUE
c Original form of norm printout
      XL2A=SQRT(XL2A)
      XL2=LOG10(XL2A)

c "Solve" for Newton correction with SGS iteration
      do 16 i=1,isgs
      call doo (2,ni,2,nj,2,nk,+1,b1,b2,b3,d,dxx)
      call doo (ni,2,nj,2,nk,2,-1,b4,b5,b6,d,dxx)
   16 continue

      DO 15 K=2,NK
      DO 15 J=2,NJ
      DO 15 I=2,NI
c Update global solution vector with Newton correction
      R(I,J,K)=R(I,J,K)+DR(I,J,K)
      RU(I,J,K)=RU(I,J,K)+DRU(I,J,K)
      RV(I,J,K)=RV(I,J,K)+DRV(I,J,K)
      RW(I,J,K)=RW(I,J,K)+DRW(I,J,K)
      E(I,J,K)=E(I,J,K)+DE(I,J,K)

c Update pressure through equation of state
      RQDQ=(RU(I,J,K)**2+RV(I,J,K)**2+
     1 RW(I,J,K)**2)/R(I,J,K)
      P(I,J,K)=GM1*(E(I,J,K)-p5*RQDQ)
   15 CONTINUE

c Apply boundary conditions
      CALL BC_UNI(xx,p,sadai,sadaj,sadak,
     &            aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)

c      if (no_output .eq. 0) then
         write(29,290) ncyc-1, xl2, PetscGetTime()-time_init
cc         write(29,290) ncyc, xl2, PetscGetTime()-time_init
#ifdef PARCH_rs6000
         call FLUSH_(29)
#else
         call flush(29)
#endif
c      endif
 290  format(1x,i5,4x,f8.4,3x,f9.2)
      if (ncyc .eq. 1) then
         xl2_init = xl2a
         if (no_output .eq. 0)
     &      write(6,*) 'xl2_init, rtol, xl2_init*rtol = ',
     &               xl2_init, rtol, xl2_init*rtol
         converge = 0
      else 
         if (no_output .eq. 0)
     &      write(6,*) 'xl2, xl2a = ',xl2,xl2a
         if (xl2a .lt. rtol*xl2_init) converge = 1
      endif

      IF (NCYC.NE.1.AND.MOD(NCYC,5).NE.0 .and. converge.ne.1) RETURN
c Compute and tabulate norms and the number of supersonic points in flowfield
      RTMAX=0.
      RTRMS=0.
      ETMAX=0.
      ETRMS=0.
      NSUP=0
      CON=GS2+RGM1

      DO 20 K=2,NK
      DO 20 J=2,NJ
      DO 20 I=2,NI
      IF (E(I,J,K).GT.P(I,J,K)*CON) NSUP=NSUP+1
      RT=DR(I,J,K)
      RTRMS=RTRMS+RT**2
      ET=DE(I,J,K)
      ETRMS=ETRMS+ET**2
      IF (ABS(RT).GT.ABS(RTMAX)) RTMAX=RT
      IF (ABS(ET).GT.ABS(ETMAX)) ETMAX=ET
   20 CONTINUE
      IF (NCYC.EQ.1) RTRMSR=RTRMS
      IF (NCYC.EQ.1) ETRMSR=ETRMS
      RTRMS=SQRT(RTRMS/RTRMSR)
      ETRMS=SQRT(ETRMS/ETRMSR)
      RTRMS=LOG10(RTRMS)
      ETRMS=LOG10(ETRMS)
      TCL=0.
      TCD=0.
      AREAS=0.
      DO 41 K=2,KTIP
      CN=0.
      CC=0.
      AREA=0.
      DO 42 I=ITL+1,ITU
      AREA=AREA+ABS(AJY(I,1,K))*.5
      PBODY=p5*(P(I,1,K)+P(I,2,K))
      CN=CN-(PBODY-RGAM)*AJY(I,1,K)
   42 CC=CC-(PBODY-RGAM)*AJX(I,1,K)
      CLS=CN*COS(-BETA)-CC*SIN(-BETA)
      CDS=CN*SIN(-BETA)+CC*COS(-BETA)
      AREAS=AREAS+AREA
      TCL=TCL+CLS
   41 TCD=TCD+CDS
      TCL=TCL/AREAS*2./FSMACH**2
      TCD=TCD/AREAS*2./FSMACH**2
      if (no_output .eq. 0) 
     & WRITE (6,200) NCYC,RTMAX,RTRMS,ETRMS,XL2,TCL,TCD,NSUP
  200 FORMAT(I5,1P6E11.3,I5)
      RETURN
      END

C***********************************************************************

      subroutine doo (i1,i2,j1,j2,k1,k2,is,a,b,c,d,dxx)
c
#include "param.h"
#include "diag.h"
#include "dvarbls.h"
#include "consts.h"
#include "gs.h"
c
c   Uniprocessor version only!
c
C   THESE DOO SUBROUTINES SOLVE LINEAR SYSTEMS USING DOOLITTLE`S METHOD.
C   FOR A REFERENCE SEE PAGE 157 OF THE BOOK ENTITLED _NUMERICAL
C   METHODS_ BY DAHLQUIST AND BJORCK.
c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /DVRBLS/ DR(NI1,NJ1,NK1),DRU(NI1,NJ1,NK1),DRV(NI1,NJ1,NK1)
c      COMMON /DVRBLS/ DRW(NI1,NJ1,NK1),DE(NI1,NJ1,NK1)
c      common /gs/ res(5,ni,nj,nk),vec(5,ni,nj,nk)

      Double  a(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      Double  b(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      Double  c(5,5,gxsf1w:xefp1,gysf1w:yefp1,gzsf1w:zefp1)
      Double  T1(D_NI),T2(D_NI),T3(D_NI),T4(D_NI),T5(D_NI)
      Double  s,z1,z2,z3,z4,z5,y1,y2,y3,y4,y5
      integer i,j,k,i1,i2,j1,j2,k1,k2,is,kc,kd,jb,jd,id,ia

      s=float(-is)
      do 2 k=k1,k2,is
      kd=k-is
      kc=k-(is+iabs(is))/2
      do 2 j=j1,j2,is
      jd=j-is
      jb=j-(is+iabs(is))/2
      do 22 i=i1,i2,is
      T1(I)=c(1,1,i,j,kc)*DR(i,j,kd)+c(1,2,i,j,kc)*DRU(i,j,kd)+
     *      c(1,3,i,j,kc)*DRV(i,j,kd)+c(1,4,i,j,kc)*DRW(i,j,kd)+
     *      c(1,5,i,j,kc)*DE(i,j,kd)
      T2(I)=c(2,1,i,j,kc)*DR(i,j,kd)+c(2,2,i,j,kc)*DRU(i,j,kd)+
     *      c(2,3,i,j,kc)*DRV(i,j,kd)+c(2,4,i,j,kc)*DRW(i,j,kd)+
     *      c(2,5,i,j,kc)*DE(i,j,kd)
      T3(I)=c(3,1,i,j,kc)*DR(i,j,kd)+c(3,2,i,j,kc)*DRU(i,j,kd)+
     *      c(3,3,i,j,kc)*DRV(i,j,kd)+c(3,4,i,j,kc)*DRW(i,j,kd)+
     *      c(3,5,i,j,kc)*DE(i,j,kd)
      T4(I)=c(4,1,i,j,kc)*DR(i,j,kd)+c(4,2,i,j,kc)*DRU(i,j,kd)+
     *      c(4,3,i,j,kc)*DRV(i,j,kd)+c(4,4,i,j,kc)*DRW(i,j,kd)+
     *      c(4,5,i,j,kc)*DE(i,j,kd)
      T5(I)=c(5,1,i,j,kc)*DR(i,j,kd)+c(5,2,i,j,kc)*DRU(i,j,kd)+
     *      c(5,3,i,j,kc)*DRV(i,j,kd)+c(5,4,i,j,kc)*DRW(i,j,kd)+
     *      c(5,5,i,j,kc)*DE(i,j,kd)
   22 CONTINUE
      do 23 i=i1,i2,is
      T1(I)=b(1,1,i,jb,k)*DR(i,jd,k)+b(1,2,i,jb,k)*DRU(i,jd,k)+
     *      b(1,3,i,jb,k)*DRV(i,jd,k)+b(1,4,i,jb,k)*DRW(i,jd,k)+
     *      b(1,5,i,jb,k)*DE(i,jd,k)+T1(I)
      T2(I)=b(2,1,i,jb,k)*DR(i,jd,k)+b(2,2,i,jb,k)*DRU(i,jd,k)+
     *      b(2,3,i,jb,k)*DRV(i,jd,k)+b(2,4,i,jb,k)*DRW(i,jd,k)+
     *      b(2,5,i,jb,k)*DE(i,jd,k)+T2(I)
      T3(I)=b(3,1,i,jb,k)*DR(i,jd,k)+b(3,2,i,jb,k)*DRU(i,jd,k)+
     *      b(3,3,i,jb,k)*DRV(i,jd,k)+b(3,4,i,jb,k)*DRW(i,jd,k)+
     *      b(3,5,i,jb,k)*DE(i,jd,k)+T3(I)
      T4(I)=b(4,1,i,jb,k)*DR(i,jd,k)+b(4,2,i,jb,k)*DRU(i,jd,k)+
     *      b(4,3,i,jb,k)*DRV(i,jd,k)+b(4,4,i,jb,k)*DRW(i,jd,k)+
     *      b(4,5,i,jb,k)*DE(i,jd,k)+T4(I)
      T5(I)=b(5,1,i,jb,k)*DR(i,jd,k)+b(5,2,i,jb,k)*DRU(i,jd,k)+
     *      b(5,3,i,jb,k)*DRV(i,jd,k)+b(5,4,i,jb,k)*DRW(i,jd,k)+
     *      b(5,5,i,jb,k)*DE(i,jd,k)+T5(I)
   23 CONTINUE
      do 2 i=i1,i2,is
      id=i-is
      ia=i-(is+iabs(is))/2
      T1(I)=s*(a(1,1,ia,j,k)*DR(id,j,k)+a(1,2,ia,j,k)*DRU(id,j,k)+
     *         a(1,3,ia,j,k)*DRV(id,j,k)+a(1,4,ia,j,k)*DRW(id,j,k)+
     *         a(1,5,ia,j,k)*DE(id,j,k)+T1(I))
      T2(I)=s*(a(2,1,ia,j,k)*DR(id,j,k)+a(2,2,ia,j,k)*DRU(id,j,k)+
     *         a(2,3,ia,j,k)*DRV(id,j,k)+a(2,4,ia,j,k)*DRW(id,j,k)+
     *         a(2,5,ia,j,k)*DE(id,j,k)+T2(I))
      T3(I)=s*(a(3,1,ia,j,k)*DR(id,j,k)+a(3,2,ia,j,k)*DRU(id,j,k)+
     *         a(3,3,ia,j,k)*DRV(id,j,k)+a(3,4,ia,j,k)*DRW(id,j,k)+
     *         a(3,5,ia,j,k)*DE(id,j,k)+T3(I))
      T4(I)=s*(a(4,1,ia,j,k)*DR(id,j,k)+a(4,2,ia,j,k)*DRU(id,j,k)+
     *         a(4,3,ia,j,k)*DRV(id,j,k)+a(4,4,ia,j,k)*DRW(id,j,k)+
     *         a(4,5,ia,j,k)*DE(id,j,k)+T4(I))
      T5(I)=s*(a(5,1,ia,j,k)*DR(id,j,k)+a(5,2,ia,j,k)*DRU(id,j,k)+
     *         a(5,3,ia,j,k)*DRV(id,j,k)+a(5,4,ia,j,k)*DRW(id,j,k)+
     *         a(5,5,ia,j,k)*DE(id,j,k)+T5(I))
      z1=res(1,i,j,k)-t1(i)-vec(1,i,j,k)
      z2=res(2,i,j,k)-t2(i)-vec(2,i,j,k)
      z3=res(3,i,j,k)-t3(i)-vec(3,i,j,k)
      z4=res(4,i,j,k)-t4(i)-vec(4,i,j,k)
      z5=res(5,i,j,k)-t5(i)-vec(5,i,j,k)
      vec(1,i,j,k)=t1(i)
      vec(2,i,j,k)=t2(i)
      vec(3,i,j,k)=t3(i)
      vec(4,i,j,k)=t4(i)
      vec(5,i,j,k)=t5(i)
      Y1=Z1
      Y2=Z2-D(2,1,I,J,K)*Y1
      Y3=Z3-D(3,1,I,J,K)*Y1-D(3,2,I,J,K)*Y2
      Y4=Z4-D(4,1,I,J,K)*Y1-D(4,2,I,J,K)*Y2-D(4,3,I,J,K)*Y3
      Y5=Z5-D(5,1,I,J,K)*Y1-D(5,2,I,J,K)*Y2-D(5,3,I,J,K)*Y3-
     * D(5,4,I,J,K)*Y4
      Z5=Y5/D(5,5,I,J,K)
      Z4=(Y4-D(4,5,I,J,K)*Z5)/D(4,4,I,J,K)
      Z3=(Y3-D(3,5,I,J,K)*Z5-D(3,4,I,J,K)*Z4)/D(3,3,I,J,K)
      Z2=(Y2-D(2,5,I,J,K)*Z5-D(2,4,I,J,K)*Z4-D(2,3,I,J,K)*Z3)
     * /D(2,2,I,J,K)
      Z1=(Y1-D(1,5,I,J,K)*Z5-D(1,4,I,J,K)*Z4-D(1,3,I,J,K)*Z3-
     * D(1,2,I,J,K)*Z2)/D(1,1,I,J,K)
      DR(I,J,K)=Z1
      DRU(I,J,K)=Z2
      DRV(I,J,K)=Z3
      DRW(I,J,K)=Z4
      DE(I,J,K)=Z5
    2 CONTINUE

      RETURN
      END

C************************************************************************

      SUBROUTINE DEQLU(d)
c
c Performs LU decomposition of the diagonal blocks
c Uniprocessor version only!
#include "param.h"
#include "diag.h"
c     COMMON /DIAG/ D(5,5,NI,NJ,NK)
      integer i,j,k

      DO 1 K=2,NK
      DO 1 J=2,NJ
      DO 1 I=2,NI
      D(2,1,I,J,K)=D(2,1,I,J,K)/D(1,1,I,J,K)
      D(2,2,I,J,K)=D(2,2,I,J,K)-D(2,1,I,J,K)*D(1,2,I,J,K)
      D(2,3,I,J,K)=D(2,3,I,J,K)-D(2,1,I,J,K)*D(1,3,I,J,K)
      D(2,4,I,J,K)=D(2,4,I,J,K)-D(2,1,I,J,K)*D(1,4,I,J,K)
      D(2,5,I,J,K)=D(2,5,I,J,K)-D(2,1,I,J,K)*D(1,5,I,J,K)
      D(3,1,I,J,K)=D(3,1,I,J,K)/D(1,1,I,J,K)
      D(3,2,I,J,K)=(D(3,2,I,J,K)-D(3,1,I,J,K)*D(1,2,I,J,K))/D(2,2,I,J,K)
      D(3,3,I,J,K)=D(3,3,I,J,K)-D(3,1,I,J,K)*D(1,3,I,J,K)-
     * D(3,2,I,J,K)*D(2,3,I,J,K)
      D(3,4,I,J,K)=D(3,4,I,J,K)-D(3,1,I,J,K)*D(1,4,I,J,K)-
     * D(3,2,I,J,K)*D(2,4,I,J,K)
      D(3,5,I,J,K)=D(3,5,I,J,K)-D(3,1,I,J,K)*D(1,5,I,J,K)-
     * D(3,2,I,J,K)*D(2,5,I,J,K)
      D(4,1,I,J,K)=D(4,1,I,J,K)/D(1,1,I,J,K)
      D(4,2,I,J,K)=(D(4,2,I,J,K)-D(4,1,I,J,K)*D(1,2,I,J,K))/D(2,2,I,J,K)
      D(4,3,I,J,K)=(D(4,3,I,J,K)-D(4,1,I,J,K)*D(1,3,I,J,K)-
     * D(4,2,I,J,K)*D(2,3,I,J,K))/D(3,3,I,J,K)
      D(4,4,I,J,K)=D(4,4,I,J,K)-D(4,1,I,J,K)*D(1,4,I,J,K)-
     * D(4,2,I,J,K)*D(2,4,I,J,K)-D(4,3,I,J,K)*D(3,4,I,J,K)
      D(4,5,I,J,K)=D(4,5,I,J,K)-D(4,1,I,J,K)*D(1,5,I,J,K)-
     * D(4,2,I,J,K)*D(2,5,I,J,K)-D(4,3,I,J,K)*D(3,5,I,J,K)
      D(5,1,I,J,K)=D(5,1,I,J,K)/D(1,1,I,J,K)
      D(5,2,I,J,K)=(D(5,2,I,J,K)-D(5,1,I,J,K)*D(1,2,I,J,K))/D(2,2,I,J,K)
      D(5,3,I,J,K)=(D(5,3,I,J,K)-D(5,1,I,J,K)*D(1,3,I,J,K)-
     * D(5,2,I,J,K)*D(2,3,I,J,K))/D(3,3,I,J,K)
      D(5,4,I,J,K)=(D(5,4,I,J,K)-D(5,1,I,J,K)*D(1,4,I,J,K)-
     * D(5,2,I,J,K)*D(2,4,I,J,K)-D(5,3,I,J,K)*D(3,4,I,J,K))
     * /D(4,4,I,J,K)
      D(5,5,I,J,K)=D(5,5,I,J,K)-D(5,1,I,J,K)*D(1,5,I,J,K)-
     * D(5,2,I,J,K)*D(2,5,I,J,K)-D(5,3,I,J,K)*D(3,5,I,J,K)-D(5,4,I,J,K)
     * *D(4,5,I,J,K)
    1 CONTINUE
      RETURN
      END
C ----------------------------------------------------------------

      integer function PVAR_UNI(xx,p,
     &                      aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                      x,y,z,pprint,tcl,tcd,tcl_scale,tcd_scale)
c
c PVAR_UNI post-processes the solution for physical quanities of interest
c Uniprocessor version only!  The parallel analog is pvar().
c
#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "coord.h"
#include "avctrs.h"
#include "ang.h"
c      COMMON /ANG/ ALPHA,BETA,PHI
c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /COORD/ X(NI,NJ,NK),Y(NI,NJ,NK),Z(NI,NJ,NK)
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /WING/ ITL,ITU,ILE,KTIP
      Double  CP2(D_NI,D_NK),S(D_NI),CL(D_NI),CD(D_NI)
      Double  ZLOC(D_NK),tcl_scale,tcd_scale
      Double  chord,pbody,tcl,tcd,areas,cn,cc,area,cls,cds
      integer i,k,pprint

      pvar_uni = 0

      DO 10 K=2,KTIP
      ZLOC(K) = (Z(1,1,K)+Z(1,1,K-1))*p5
      CHORD   = X(ITU,1,K)-X(ILE,1,K)
      DO 10 I=ITL,ITU+1
      PBODY    = p5*(P(I,1,K)+P(I,2,K))
      CP2(I,K) = (PBODY-RGAM)/FSMACH**2*2.
   10 S(I)     = ((X(I,1,K)+X(I-1,1,K))*p5-X(ILE,1,K))/CHORD
      if (pprint .eq. 1) then
      IF (NI.EQ.49) WRITE (6,20) NCYC
   20 FORMAT (/,4X,'WING SURFACE PRESSURE COEFFICIENTS FOR',
     * ' NCYC =',I6,/,4X,'X/C',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP')
      IF (NI.EQ.49) WRITE (6,21) (ZLOC(K),K=2,KTIP)
   21 FORMAT (4X,'Z = ',5F8.4,/)
      IF (NI.EQ.49)
     * WRITE (6,30) (S(I),(CP2(I,K),K=2,KTIP),I=ITL,ITU+1)
   30 FORMAT(6F8.4)
      IF (NI.EQ.97) WRITE (6,90) NCYC
   90 FORMAT (/,4X,'WING SURFACE PRESSURE COEFFICIENTS FOR',
     * ' NCYC =',I6,/,4X,'X/C',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP')
      IF (NI.EQ.97) WRITE (6,91) (ZLOC(K),K=2,KTIP)
   91 FORMAT (4X,'Z = ',10F8.4,/)
      IF (NI.EQ.97)
     * WRITE (6,93) (S(I),(CP2(I,K),K=2,KTIP),I=ITL,ITU+1)
   93 FORMAT(11F8.4)
      endif
      TCL=0.
      TCD=0.
      AREAS=0.
      DO 41 K=2,KTIP
      CN=0.
      CC=0.
      AREA=0.
      DO 42 I=ITL+1,ITU
      AREA  = AREA+ABS(AJY(I,1,K))*.5
      CN    = CN-CP2(I,K)*AJY(I,1,K)
   42 CC    = CC-CP2(I,K)*AJX(I,1,K)
      CLS   = CN*COS(-BETA)-CC*SIN(-BETA)
      CDS   = CN*SIN(-BETA)+CC*COS(-BETA)
      CL(K) = CLS/AREA
      CD(K) = CDS/AREA
      AREAS = AREAS+AREA
      TCL   = TCL+CLS
   41 TCD   = TCD+CDS
      TCL   = TCL/AREAS
      TCD   = TCD/AREAS
      TCL_scale = TCL/AREAS*2./FSMACH**2
      TCD_scale = TCD/AREAS*2./FSMACH**2
      WRITE (6,43)
   43 FORMAT (/ /,20X,'SPANWISE LIFT DISTRIBUTION',/ /)
      WRITE (6,44)
   44 FORMAT (5X,'Z',12X,'CL',12X,'CD',/)
      WRITE (6,45) (ZLOC(K),CL(K),CD(K),K=2,KTIP)
   45 FORMAT (2X,F6.3,4X,F10.4,4X,F10.4)
      WRITE (6,46) TCL,TCD
   46 FORMAT (/,2X,'WING LIFT COEFFICIENT =',F10.4,5X,
     1 'WING DRAG COEFFICIENT =',F10.4,/)

      RETURN
      END
