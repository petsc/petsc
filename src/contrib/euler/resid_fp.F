c
c  "$Id: resid_fp.F,v 1.15 1997/10/17 19:46:42 curfman Exp keyes $";
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
     &                dxx,den,xvel,yvel,zvel,x,y,z)

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
#include "coord.h"

      integer first_time_resid,ierr,resid_fp,residbc_fp
  
      localfortfct_fp = 0

C  If using implicit BC's, compute function components for boundary points 
      ierr = residbc_fp(xx, fff, p, xx_bc, p_bc, sadai, sadaj, sadak,
     &         aix, ajx, akx, aiy, ajy, aky, aiz, ajz, akz,
     &         den, xvel, yvel, zvel, x, y, z)
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
     &         den, xvel, yvel, zvel, x, y, z)
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
     &                  den,xvel,yvel,zvel,x,y,z)

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
#include "coord.h"

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
      double precision dx, dy, xv, yv, zv, dn 
      double precision xaux0, xauxm, xauxp, yaux0, yauxm, yauxp
      resid_fp = 0

c set dx and dy from "interior" area metrics (assumes that dz spacing
c is uniformly unity, and that mesh is uniform

      dx = ajy(xsf2,ysf2,zsf2) 
      dy = aix(xsf2,ysf2,zsf2)
c      print *, 'mesh spacings', dx, dy

c
c  Uniform grid auxiliaries (not needed now)
c
c      do 1 k=gzsf1,gzef01
c      do 1 j=gysf1,gyef01
c      do 1 i=gxsf1,gxef01
c	  xvel(i,j,k)= p5* ( fp(i+1,j,k)-fp(i,j,k) +
c     &                       fp(i+1,j+1,k)-fp(i,j+1,k) )/dx
c	  yvel(i,j,k)= p5* ( fp(i,j+1,k)-fp(i,j,k) +
c     &                       fp(i+1,j+1,k)-fp(i+1,j,k) )/dy
c          zvel(i,j,k)= zero
c          den (i,j,k)= ( one+p5*gm1*(fsmach**2-
c     &                 (xvel(i,j,k)**2+yvel(i,j,k)**2)) )**rgm1
c1     continue
c
c velocities at (i,j) interior points, for all k:
c
      do 100 k=gzsf1,gzef01
      do 100 j=gysf2,gyefm1
      do 100 i=gxsf2,gxefm1
c       xne = p25 * (x(i,j,k)+x(i+1,j,k)+x(i,j+1,k)+x(i+1,j+1,k))
c	xnw = p25 * (x(i,j,k)+x(i-1,j,k)+x(i,j+1,k)+x(i-1,j+1,k))
c	xse = p25 * (x(i,j,k)+x(i+1,j,k)+x(i,j-1,k)+x(i+1,j-1,k))
c	xsw = p25 * (x(i,j,k)+x(i-1,j,k)+x(i,j-1,k)+x(i-1,j-1,k))
c	yne = p25 * (y(i,j,k)+y(i+1,j,k)+y(i,j+1,k)+y(i+1,j+1,k))
c	ynw = p25 * (y(i,j,k)+y(i-1,j,k)+y(i,j+1,k)+y(i-1,j+1,k))
c	yse = p25 * (y(i,j,k)+y(i+1,j,k)+y(i,j-1,k)+y(i+1,j-1,k))
c	ysw = p25 * (y(i,j,k)+y(i-1,j,k)+y(i,j-1,k)+y(i-1,j-1,k))
c	xe  = p5 * (xne + xse)
c	xw  = p5 * (xnw + xsw)
c	yn  = p5 * (yne + ynw)
c	ys  = p5 * (yse + ysw)
c       cen = (y(i,j,k)-yse) / (yne-yse)
c	ces = one - cen
c       cwn = (y(i,j,k)-ysw) / (ynw-ysw)
c	cws = one - cen
c       cne = (x(i,j,k)-xnw) / (xne-xnw)
c	cnw = one - cne
c       cse = (x(i,j,k)-xsw) / (xse-xsw)
c	csw = one - cse
c       fpe = cen*fp(i+1,j+1,k) + ces*fp(i+1,j,k)
c       fpw = cwn*fp(i,j+1,k)   + cws*fp(i,j,k)
c       fpn = cne*fp(i+1,j+1,k) + cnw*fp(i,j+1,k)
c       fps = cse*fp(i+1,j,k)   + csw*fp(i,j,k)
c       xvel(i,j,k) = (fpe-fpw) / (xe-xw)
c       yvel(i,j,k) = (fpn-fps) / (yn-ys)
	call velfcn(x(i-1,j-1,k),x(i,j-1,k),x(i+1,j-1,k),
     &              x(i-1,j  ,k),x(i,j  ,k),x(i+1,j  ,k),
     &              x(i-1,j+1,k),x(i,j+1,k),x(i+1,j+1,k),
     &              y(i-1,j-1,k),y(i,j-1,k),y(i+1,j-1,k),
     &              y(i-1,j  ,k),y(i,j  ,k),y(i+1,j  ,k),
     &              y(i-1,j+1,k),y(i,j+1,k),y(i+1,j+1,k),
     &              fp(i,j,k),fp(i+1,j,k),
     &              fp(i,j+1,k),fp(i+1,j+1,k),
     &  	    xvel(i,j,k), yvel(i,j,k),
     &              x(i,j,k),y(i,j,k))
        zvel(i,j,k) = zero
        den (i,j,k) = ( one+p5*gm1*(fsmach**2-
     &               (xvel(i,j,k)**2+yvel(i,j,k)**2)) )**rgm1
100   continue
c
c velocities at left edge dual points (includes upper and lower corners):
c
      i=gxsf1
      do 101 k=gzsf1,gzef01
      do 101 j=gysf1,gyef01
	xaux0 = two * x(i,j,k) - x(i+1,j,k)
	yaux0 = two * y(i,j,k) - y(i+1,j,k)
	call velfcn(xaux0       ,x(i,j-1,k),x(i+1,j-1,k),
     &              xaux0       ,x(i,j  ,k),x(i+1,j  ,k),
     &              xaux0       ,x(i,j+1,k),x(i+1,j+1,k),
     &              yaux0       ,y(i,j-1,k),y(i+1,j-1,k),
     &              yaux0       ,y(i,j  ,k),y(i+1,j  ,k),
     &              yaux0       ,y(i,j+1,k),y(i+1,j+1,k),
     &              fp(i,j,k),fp(i+1,j,k),
     &              fp(i,j+1,k),fp(i+1,j+1,k),
     &  	    xvel(i,j,k), yvel(i,j,k),
     &              x(i,j,k),y(i,j,k))
        zvel(i,j,k) = zero
        den (i,j,k) = ( one+p5*gm1*(fsmach**2-
     &               (xvel(i,j,k)**2+yvel(i,j,k)**2)) )**rgm1
101   continue

c
c velocities at right edge dual points (includes upper and lower corners):
c
      i=gxef01
      do 102 k=gzsf1,gzef01
      do 102 j=gysf1,gyef01
	xaux0 = two * x(i,j,k) - x(i-1,j,k)
	yaux0 = two * y(i,j,k) - y(i-1,j,k)
	call velfcn(x(i-1,j-1,k),x(i,j-1,k),xaux0       ,
     &              x(i-1,j  ,k),x(i,j  ,k),xaux0       ,
     &              x(i-1,j+1,k),x(i,j+1,k),xaux0       ,
     &              y(i-1,j-1,k),y(i,j-1,k),yaux0       ,
     &              y(i-1,j  ,k),y(i,j  ,k),yaux0       ,
     &              y(i-1,j+1,k),y(i,j+1,k),yaux0       ,
     &              fp(i,j,k),fp(i+1,j,k),
     &              fp(i,j+1,k),fp(i+1,j+1,k),
     &  	    xvel(i,j,k), yvel(i,j,k),
     &              x(i,j,k),y(i,j,k))
        zvel(i,j,k) = zero
        den (i,j,k) = ( one+p5*gm1*(fsmach**2-
     &               (xvel(i,j,k)**2+yvel(i,j,k)**2)) )**rgm1
102   continue

c
c velocities at bottom edge dual points:
c
      j=gysf1
      do 103 k=gzsf1,gzef01
      do 103 i=gxsf2,gxefm1
	xaux0 = two * x(i  ,j,k) - x(i  ,j+1,k)
	xauxm = two * x(i-1,j,k) - x(i-1,j+1,k)
	xauxp = two * x(i+1,j,k) - x(i+1,j+1,k)
	yaux0 = two * y(i  ,j,k) - y(i  ,j+1,k)
	yauxm = two * y(i-1,j,k) - y(i-1,j+1,k)
	yauxp = two * y(i+1,j,k) - y(i+1,j+1,k)
	call velfcn(xauxm       ,xaux0     ,xauxp       ,
     &              x(i-1,j  ,k),x(i,j  ,k),x(i+1,j  ,k),
     &              x(i-1,j+1,k),x(i,j+1,k),x(i+1,j+1,k),
     &              yauxm       ,yaux0     ,yauxp       ,
     &              y(i-1,j  ,k),y(i,j  ,k),y(i+1,j  ,k),
     &              y(i-1,j+1,k),y(i,j+1,k),y(i+1,j+1,k),
     &              fp(i,j,k),fp(i+1,j,k),
     &              fp(i,j+1,k),fp(i+1,j+1,k),
     &  	    xvel(i,j,k), yvel(i,j,k),
     &              x(i,j,k),y(i,j,k))
        zvel(i,j,k) = zero
        den (i,j,k) = ( one+p5*gm1*(fsmach**2-
     &               (xvel(i,j,k)**2+yvel(i,j,k)**2)) )**rgm1
103   continue

c
c velocities at top edge dual points:
c
      j=gyef01
      do 104 k=gzsf1,gzef01
      do 104 i=gxsf2,gxefm1
	xaux0 = two * x(i  ,j,k) - x(i  ,j-1,k)
	xauxm = two * x(i-1,j,k) - x(i-1,j-1,k)
	xauxp = two * x(i+1,j,k) - x(i+1,j-1,k)
	yaux0 = two * y(i  ,j,k) - y(i  ,j-1,k)
	yauxm = two * y(i-1,j,k) - y(i-1,j-1,k)
	yauxp = two * y(i+1,j,k) - y(i+1,j-1,k)
	call velfcn(x(i-1,j-1,k),x(i,j-1,k),x(i+1,j-1,k),
     &              x(i-1,j  ,k),x(i,j  ,k),x(i+1,j  ,k),
     &              xauxm       ,xaux0     ,xauxp       ,
     &              y(i-1,j-1,k),y(i,j-1,k),y(i+1,j-1,k),
     &              y(i-1,j  ,k),y(i,j  ,k),y(i+1,j  ,k),
     &              yauxm       ,yaux0     ,yauxp       ,
     &              fp(i,j,k),fp(i+1,j,k),
     &              fp(i,j+1,k),fp(i+1,j+1,k),
     &  	    xvel(i,j,k), yvel(i,j,k),
     &              x(i,j,k),y(i,j,k))
        zvel(i,j,k) = zero
        den (i,j,k) = ( one+p5*gm1*(fsmach**2-
     &               (xvel(i,j,k)**2+yvel(i,j,k)**2)) )**rgm1
104   continue

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
      xv = p5 * (xvel(i,j,k)+xvel(i,j-1,k))
      yv = p5 * (yvel(i,j,k)+yvel(i,j-1,k))
      zv = p5 * (zvel(i,j,k)+zvel(i,j-1,k))
      dn = p5 * (den(i,j,k) +den(i,j-1,k) )
c
c   Form i-flux at the FV cell edge centers (I+1/2)
c
      F(I) = ( xv*aix(i,j,k) +
     &         yv*aiy(i,j,k) +
     &         zv*aiz(i,j,k) ) * dn
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
      xv = p5 * (xvel(i,j,k)+xvel(i-1,j,k))
      yv = p5 * (yvel(i,j,k)+yvel(i-1,j,k))
      zv = p5 * (zvel(i,j,k)+zvel(i-1,j,k))
      dn = p5 * (den(i,j,k) +den(i-1,j,k) )
c
c   Form j-flux at the FV cell edge centers (J+1/2)
c
      F(J) = ( xv*ajx(i,j,k) +
     &         yv*ajy(i,j,k) +
     &         zv*ajz(i,j,k) ) * dn
   31 continue
c
c   Form j-contribution to residual at cell centers (I,J,K) 
c
      DO 3 J=ysf2,yef01
      DFP(I,J,K) = DFP(I,J,K) + F(J)-F(J-1)
    3 CONTINUE

c
c   Supply the velocity and density auxiliaries at the (1,1)-corner
c   (in FORTRAN indexing convention), for graphical output
c
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
c=======================================================================
	subroutine velfcn(xmm,xim,xpm,xmj,xij,xpj,xmp,xip,xpp,
     &                    ymm,yim,ypm,ymj,yij,ypj,ymp,yip,ypp,
     &                    fij,fpj,fip,fpp,xvel,yvel,xctr,yctr)

	double precision xmm,xim,xpm,xmj,xij,xpj,xmp,xip,xpp,
     &                   ymm,yim,ypm,ymj,yij,ypj,ymp,yip,ypp,
     &                   fij,fpj,fip,fpp, xvel,yvel, p25,
     &			 xne,xnw,xse,xsw,yne,ynw,yse,ysw,
     &                   cen,ces,cwn,cws,cne,cnw,cse,csw,
     &                   xe,xw,yn,ys,fpe,fpw,fpn,fps,xctr,yctr

	parameter (p25=0.25d0,p5=0.5d0,one=1.0d0)
	xne = p25 * (xij+xpj+xip+xpp)
	xnw = p25 * (xij+xmj+xip+xmp)
	xse = p25 * (xij+xpj+xim+xpm)
	xsw = p25 * (xij+xmj+xim+xmm)
	yne = p25 * (yij+ypj+yip+ypp)
	ynw = p25 * (yij+ymj+yip+ymp)
	yse = p25 * (yij+ypj+yim+ypm)
	ysw = p25 * (yij+ymj+yim+ymm)
	xe  = p5 * (xne + xse)
	xw  = p5 * (xnw + xsw)
	yn  = p5 * (yne + ynw)
	ys  = p5 * (yse + ysw)
        cen = (yctr-yse) / (yne-yse)
	ces = one - cen
        cwn = (yctr-ysw) / (ynw-ysw)
	cws = one - cen
        cne = (xctr-xnw) / (xne-xnw)
	cnw = one - cne
        cse = (xctr-xsw) / (xse-xsw)
	csw = one - cse
        fpe = cen*fpp + ces*fpj
        fpw = cwn*fip + cws*fij
        fpn = cne*fpp + cnw*fip
        fps = cse*fpj + csw*fij
        xvel = (fpe-fpw) / (xe-xw)
        yvel = (fpn-fps) / (yn-ys)
        return
	end
c ---------------------------------------------------------------

      integer function residbc_fp(xx,fff,p,xx_bc,p_bc,
     &                sadai,sadaj,sadak,
     &                aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                den,xvel,yvel,zvel,x,y,z)

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
#include "coord.h"

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

      double precision xloc
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
c      IF (I.LE.ILE.OR.I.GT.ITU) GO TO 4
       IF (I.GT.0)  GO TO 4
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
     &             dy*fsmach*pi*sin(pi*xloc)*cos(pi*xloc)
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
c      xloc=-1.50d0+3.0d0*(dfloat(i-1)-p5)/dfloat(ni-1)
c      fbcfpj2(i,k)=fp(i,nj1,k)-xloc*fsmach
c inhomogeneous Neumann 
      fbcfpj2(i,k)=fp(i,nj1,k)-fp(i,nj,k)
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
c      fbcfpi2(j,k)=fp(ni1,j,k)-fp(ni,j,k)-
c     &    dx*fsmach
      fbcfpi2(j,k)=fp(ni1,j,k)
     &   -two*fp(ni,j,k)+fp(ni-1,j,k)
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
