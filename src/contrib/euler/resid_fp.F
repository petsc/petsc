c
c  "$Id: resid_fp.F,v 1.9 1997/10/16 13:49:29 keyes Exp curfman $";
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

      integer first_time_resid,ierr,resid_fp,residbc_fp
  
      localfortfct_fp = 0

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
      double precision p25 
      parameter(p25=0.25d0)
      double precision dx, dy

      resid_fp = 0

c set dx and dy from "interior" area metrics (assumes that dz spacing
c is uniformly unity, and that mesh is uniform

      dx = ajy(xsf2,ysf2,zsf2) 
      dy = aix(xsf2,ysf2,zsf2)
c      print *, 'mesh spacings', dx, dy

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
      xvel(i,j,k)=    (fp(i+1,j,k)-fp(i,j,k))/dx
      yvel(i,j,k)=p25*(fp(i+1,j+1,k)-fp(i+1,j-1,k)+
     &                 fp(i,j+1,k)-fp(i,j-1,k))/dy
      zvel(i,j,k)=zero
c set density to 1.0 for debugging
c      den (i,j,k)=one
      den (i,j,k)=( one+p5*gm1*(fsmach**2-
     &              (xvel(i,j,k)**2+yvel(i,j,k)**2)) )**rgm1
c
c   Form i-flux at the FV cell edge centers (I+1/2)
c
      F(I) = ( xvel(i,j,k)*aix(i,j,k) +
     &         yvel(i,j,k)*aiy(i,j,k) +
     &         zvel(i,j,k)*aiz(i,j,k) ) * den(i,j,k)
   21 continue
c
c   Form i-contribution to residual at cell centers (I,J,K) 
c
      DO 2 I=xsf2,xef01
      DFP(I,J,K)=F(I)-F(I-1)
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
      xvel(i,j,k)=p25*(fp(i+1,j+1,k)-fp(i-1,j+1,k)+
     &                 fp(i+1,j,k)-fp(i-1,j,k))/dx
      yvel(i,j,k)=    (fp(i,j+1,k)-fp(i,j,k))/dy
      zvel(i,j,k)=zero
c set density to 1.0 for debugging
c      den (i,j,k)=one
      den (i,j,k)=( one+p5*gm1*(fsmach**2-
     &              (xvel(i,j,k)**2+yvel(i,j,k)**2)) )**rgm1
c
c   Form j-flux at the FV cell edge centers (J+1/2)
c
      F(J) = ( xvel(i,j,k)*ajx(i,j,k) +
     &         yvel(i,j,k)*ajy(i,j,k) +
     &         zvel(i,j,k)*ajz(i,j,k) ) * den(i,j,k)
   31 continue
c
c   Form j-contribution to residual at cell centers (I,J,K) 
c
      DO 3 J=ysf2,yef01
      DFP(I,J,K) = DFP(I,J,K) + F(J)-F(J-1)
    3 CONTINUE

C   THE FOLLOWING IS FOR 2-D PLANAR FLOW
      RETURN
C   FLUX IN THE K DIRECTION
c      do 4 j=ysf2,yef01
c      do 4 i=xsf2,xef01
c      do 41 k=gzsf1,gzefw
c      DO 4 J=2,NJ
c      DO 4 I=2,NI
c      DO 41 K=1,NK
c
c   Form density and contravariant velocities at FV cell edge centers (K+1/2)
c   as functions of PHI(I,J,K) and PHI(I,J,K+1)
c
c       xvel(i,j,k)=...
c       yvel(i,j,k)=...
c       zvel(i,j,k)=...
c       den(i,j,k)=...
c
c   Form k-flux at the FV cell edge centers (K+1/2)
c
c      F(K) = ( xvel(i,j,k)*akx(i,j,k) +
c     *         yvel(i,j,k)*aky(i,j,k) +
c     *         zvel(i,j,k)*akz(i,j,k) ) * den(i,j,k)
c   41 continue
c
c   Form k-contribution to residual at cell centers (I,J,K) 
c
c      DO 4 K=zsf2,zef01
c      DFP(I,J,K) = DFP(I,J,K) + F(K)-F(K-1)
c    4 CONTINUE
c
c      RETURN
      END

c ---------------------------------------------------------------

      integer function residbc_fp(xx,fff,p,xx_bc,p_bc,
     &                sadai,sadaj,sadak,
     &                aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                den,xvel,yvel,zvel)

c residbc - Forms the residual components due to boundary conditions.
c           This routine is called by localfortranfct().
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
      double precision  pijnk1,rni1jk,r1jk,xloc
      integer i,j,k

      double precision dx, dy

      residbc_fp = 0

c set dx and dy from "interior" area metrics (assumes that dz spacing
c is uniformly unity, and that mesh is uniform)

      dx = ajy(xsf2,ysf2,zsf2) 
      dy = aix(xsf2,ysf2,zsf2)
c      print *, bump
c
c bottom of duct
      if (ysf1 .eq. 1) then
        do 2 k=zsf1,zefp1
        do 2 i=xsf1,xefp1
c        do 2 k=zsf2,zef01
c        do 2 i=xsf2,xef01
c      DO 2 K=2,NK
c      DO 2 I=2,NI
      IF (I.LE.ILE.OR.I.GT.ITU) GO TO 4
c
c inhomogeneous Neumann for FP over the "bump"
c
c                             ILE=17            ITU=33
c    |<---- domain ...        |<---- BUMP ----->|       ... domain ---->|
c    x1    x2    x3           x17               x33         x47   x48   x49
c o--|--o--|--o--| ...  |--o--|--o--| ... |--o--|--o--| ... |--o--|--o--|--o
c p1    p2    p3           p17   p18         p33   p34         p48   p49   p50
c
      xloc=-1.50d0+3.0d0*(dfloat(i-1)-p5)/dfloat(ni-1)
      fbcfpj1(i,k)=fp(i,1,k)-fp(i,2,k)+bump*
     &             dy*fsmach*sin(pi*xloc)*cos(pi*xloc)
      GO TO 2
    4 CONTINUE
c
c homogeneous Neumann for FP on either side of the "bump"
c
      fbcfpj1(i,k)=fp(i,1,k)-fp(i,2,k)
    2 CONTINUE
      endif
c
c top of duct
c
      if (yef01 .eq. nj) then
        do 20 k=zsf1,zefp1
        do 20 i=xsf1,xefp1
c        do 20 k=zsf2,zef01
c        do 20 i=xsf2,xef01
c      DO 20 K=2,NK
c      DO 20 I=2,NI
c
c inhomogeneous Dirichlet 
      xloc=-1.50d0+3.0d0*(dfloat(i-1)-p5)/dfloat(ni-1)
      fbcfpj2(i,k)=fp(i,nj1,k)-xloc*fsmach
   20 CONTINUE
      endif

c
c zmin boundary is homogeneous Neumann
      if (zsf1 .eq. 1) then
        do 6 j=ysf2,yef01
        do 6 i=xsf2,xef01
c      DO 6 J=2,NJ
c      DO 6 I=2,NI
      fbcfpk1(i,j)=fp(i,j,1)-fp(i,j,2)
    6 CONTINUE
      endif
c
c zmax boundary is homogeneous Neumann
c      if (zef01 .eq. nk) then
      if (zef01 .eq. nk_boundary) then
        do 7 j=ysf2,yef01
        do 7 i=xsf2,xef01
       fbcfpk2(i,j)=
     &   fp(i,j,zefp1)-fp(i,j,zef01)
        if (fbcfpk2(i,j) .ne. 0.0) 
     &   write(6,*) 'j,i,f = ',j,i,fbcfpk2(i,j)
    7 CONTINUE
      endif
c
c end of duct
c
      if (xef01 .eq. ni) then
        do 8 k=zsf1,zefp1
        do 8 j=ysf2,yef01
c        do 8 k=zsf2,zef01
c        do 8 j=ysf2,yef01
c      DO 8 K=2,NK
c      DO 8 J=2,NJ
c
c simple outflow condition
      fbcfpi2(j,k)=fp(ni1,j,k)-fp(ni,j,k)-
     &             dx*fsmach
    8 CONTINUE
      endif
c
c inflow of duct
c
      if (xsf1 .eq. 1) then
      xloc=-1.50d0+3.0d0*(-p5)/dfloat(ni-1)
        do 9 k=zsf1,zefp1
        do 9 j=ysf2,yef01
c       do 9 k=zsf2,zef01
c       do 9 j=ysf2,yef01
      fbcfpi1(j,k)=fp(1,j,k)-xloc*fsmach
    9 CONTINUE
      endif

      RETURN
      END
