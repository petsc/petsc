c  $Id: user.F,v 1.19 2000/04/30 17:00:00 kaushik Exp kaushik $
c---------------------------------------------------------------
c The following subroutines are from node2t.f in the original
c code - D. K. Kaushik (1/17/97)
c---------------------------------------------------------------
c
c
c 2-D Navier Stokes on Unstructured Grids

c============================== FORLINK ==============================72
c
c  FORLINK establishes links between FORTRAN common blocks and C
c
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "FORLINK"
      subroutine FORLINK()

      implicit none
#include "include/finclude/petscdef.h"
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      Scalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,
     &        nstage,ncyct,iramp,nitfo
      Scalar  gtol
      integer icycle,nsrch,ilu0,ifcn
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo
      common/gmcom/gtol,icycle,nsrch,ilu0,ifcn

      call CLINK(title,cfl1,gtol)
c
c End of subroutine FORLINK
c
      return
      end
c============================== Block_Initialization  ======== =======72
c
c Initializes the common blocks members for turbulence model
c
c=====================================================================72
      block data Block_Initialization
      implicit none
#include "include/finclude/petscdef.h"
      Scalar vkar,cmu,ce1,ce2,aplus1,aplus2,turbinf
      Scalar cb1,sig,cb2,cw1,cw2,cw3,cv1,ct1,ct2,ct3,ct4
      common/turb/vkar,cmu,ce1,ce2,aplus1,aplus2,turbinf
      common/spalrt/cb1,sig,cb2,cw1,cw2,cw3,cv1,ct1,ct2,ct3,ct4
      data vkar,cmu,ce1,ce2/0.41,0.09,1.2,2.0 /
      data aplus1,aplus2,turbinf/26.0,10.0,0.1 /
      data cb1,sig,cb2,cw2,cw3/0.1355,0.66667,0.622,0.3,2.0/
c
c Comment out old coefficients
c      data cv1,ct1,ct2,ct3,ct4/7.1,1.0,2.0,1.1,2.0/
c
       data cv1,ct1,ct2,ct3,ct4/7.1d0,1.0d0,2.0d0,1.2d0,0.5d0/

      end

c================================== INIT =============================72
c
c Initializes the flow field
c
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "INIT"
      subroutine INIT(nnodes, qvec, turbre, amut,nvnode, ivnode,irank)
      implicit none
#include "include/finclude/petscdef.h"
      Scalar  turbre(1),amut(1)
      integer ivnode(1),nnodes,nvnode,irank
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      Scalar  gamma,gm1,gp1,gm1g,gp1g,ggm1
      Scalar  p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      Scalar  vkar,cmu,ce1,ce2,aplus1,aplus2,turbinf
      Scalar  cb1,sig,cb2,cw1,cw2,cw3,cv1,ct1,ct2,ct3,ct4
      Scalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,
     &        nstage,ncyct,iramp,nitfo
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/turb/vkar,cmu,ce1,ce2,aplus1,aplus2,turbinf
      common/spalrt/cb1,sig,cb2,cw1,cw2,cw3,cv1,ct1,ct2,ct3,ct4
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo
      Scalar  pi,conv,rmu,chi,fv1
      Integer i,k,n
#if defined(INTERLACING)
       Scalar qvec(4,nnodes)
#define qnode(i,j) qvec(i,j)
#else
       Scalar qvec(nnodes,4)
#define qnode(i,j) qvec(j,i)
#endif
c
      cw1 = cb1/vkar/vkar + (1.0d0 + cb2)/sig
      if(ivisc.gt.4d0)turbinf = 1.341946d0
c      if(ivisc.gt.4)turbinf = 0.5
      if(ivisc.gt.4.and.itran.eq.1)turbinf = 0.1d0
c
c Note that for Spalarts model, I use turbinf as the freestream value of
c the dependent variable just as in the Baldwin-Barth model.
c The constant is set so that in the freestream, nu_t=0.009 (1.341946)
c 
c     print *, "I am in INIT"
      res0 = 1.0d0
      resc = 1.0d0

      gamma = 1.0d0
      gm1   = 1.0d0
      gp1   = 1.0d0
      gm1g  = 1.0d0
      gp1g  = 1.0d0
      ggm1  = 1.0d0
 
      pi = 4.0d0*datan(1.0d0)
      conv = 180.0d0/pi
      p0    = 1.0d0
#if defined(CFL3D_AXIS)
      u0    = cos(alpha/conv)
      v0    = 0.0d0
      w0    = sin(alpha/conv)
#else
      u0    = cos(alpha/conv)
      v0    = sin(alpha/conv)
      w0    = 0.0d0
#endif
      do n = 1,nnodes
         qnode(1,n) = p0
         qnode(2,n) = u0
         qnode(3,n) = v0
         qnode(4,n) = w0
      enddo
      if (ivisc.eq.3) then
       do n = 1,nnodes
         turbre(n)  = 0.0d0
         amut(n)    = 0.0d0
       enddo
      endif
c
c If viscous, zero out the velocity on the surface
c
c     print *, "Just Before Viscous"

      do 9010 i = 1,nvnode
c
c Compute the velocity normal to the surface
c
        k       = ivnode(i)
        qnode(2,k) = 0.0d0
        qnode(3,k) = 0.0d0
        qnode(4,k) = 0.0d0
 9010 continue
c
c If turbulent, initialize turbre
c
      if (ivisc.eq.3.or.ivisc.eq.4) then
         if (irank .eq. 0) write(10,110)
         if (irank .eq. 0) write(10,120)vkar,cmu,ce1,ce2
         if (irank .eq. 0) write(10,130)aplus1,aplus2,turbinf
         do 1010 n = 1,nnodes
            turbre(n) = turbinf
            amut(n)   = cmu*turbinf
 1010    continue
      end if
 
      if (ivisc.eq.5.or.ivisc.eq.6) then
         if (irank .eq. 0) write(10,110)
         if (irank .eq. 0) write(10,140)vkar,cb1,sig,cb2
         if (irank .eq. 0) write(10,150)cw1,cw2,cw3,cv1
         if (irank .eq. 0) write(10,160)ct1,ct2,ct3,ct4
         do 1020 n = 1,nnodes
            turbre(n) = turbinf
            rmu = 1.0d0
            chi = turbre(n)/rmu
            fv1 = chi**3/(chi**3 + cv1**3)
            amut(n)   = fv1*turbinf
 1020    continue
      end if
c     print *, "I am out of INIT"
 
      return
  110 format(1h ,'Parameters for turbulence model')
  120 format(1h ,'k=',f10.5,' cmu=',f10.5,' ce1=',f10.5,'ce2=',f10.5)
  130 format(1h ,'aplus1',f10.5,' aplus2=',f10.5,' turbinf=',f10.5)
  140 format(1h ,'k=',f10.5,' cb1=',f10.5,' sig=',f10.5,'cb2=',f10.5)
  150 format(1h ,'cw1=',f10.5,' cw2=',f10.5,' cw3=',f10.5,' cv1=',f10.5)
  160 format(1h ,'ct1=',f10.5,' ct2=',f10.5,' ct3=',f10.5,' ct4=',f10.5)
c
c End of subroutine INIT
c
      
      end


c================================ READR1 ====================================
c
c  Reads input parameters
c
c============================================================================
#undef __FUNC__
#define __FUNC__ "READR1"
      subroutine READR1(ileast, irank)

      implicit none
#include "include/finclude/petscdef.h"
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      Scalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,
     &        nstage,ncyct,iramp,nitfo
      Scalar  gtol
      integer icycle,nsrch,ilu0,ifcn
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo
      common/gmcom/gtol,icycle,nsrch,ilu0,ifcn
      integer ileast, irank
      integer i

      read(7,10)(title(i),i=1,20)
      if (irank .eq. 0) write(10,11)(title(i),i=1,20)

      read(7,10)

      read(7,24)mseq,ihane,ivisc,ileast,iflim,jupdate,ifcn
      if (irank .eq. 0) write(10,25)mseq,ihane,ivisc
      if (irank .eq. 0) write(10,28)ileast,iflim,jupdate

      read(7,10)

      read(7,12)beta,alpha,Re
      if (irank .eq. 0) write(10,13)beta,alpha,Re

      read(7,10)

      read(7,26)cfl2,dt,irest,itran,nbtran
      if (irank .eq. 0) write(10,27)cfl2,dt,irest

      if (irank .eq. 0) then
       if(ivisc.eq.5.or.ivisc.eq.6) write(10,123)itran,nbtran
      endif

   10 format(20a4)
   11 format(1h ,20a4)
   12 format(2f10.5,e14.7)
   13 format(1h ,'beta = ',f10.5,' Alpha = ',f10.5,' Re = ',e14.7)
   24 format(i10,i10,i10,i10,i10,i10,i10)
   25 format(1h ,'mseq = ',i3,' ihane = ',i3,' ivisc=',i3)
   26 format(2f10.5,3i10)
   27 format(1h ,' cfl2= ',e14.7,' dt= ',f10.5,'irest= ',i5)
   28 format(1h ,'ileast= ',i5,' iflim= ',i5,' jupdate= ',i5)
  123 format(1h ,'itran = ',i5,' nbtran= ',i5)

      return
      end


c================================ RDGPAR =============================72
c
c Reads grid parameters
c I am using this subroutine in modified form - DKK (1/8/97)
c
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "RDGPAR"
      subroutine RDGPAR(nnodes,ncell,nedge,
     &                  ncolort,ncolore,
     &                  nnbound,nvbound,nfbound,
     &                  nnfacet,nvfacet,nffacet,
     &                  nsnode,nvnode,nfnode,ntte,
     &                  nsface,nvface,nfface,
     &                  irank)

      implicit none
#include "include/finclude/petscdef.h"
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      integer nnodes,ncell,nedge, 
     &        ncolort,ncolore,
     &        nnbound,nvbound,nfbound,
     &        nnfacet,nvfacet,nffacet,
     &        nsnode,nvnode,nfnode,ntte,
     &        nsface,nvface,nfface,
     &        irank

      character*40 filename

      integer unit
c 
c  DEBUG: to check things out
c  
      nsface = 0
      nvface = 0
      nfface = 0
c
c  end DEBUG: to check things out
c

      unit =  20

c     read(7,*) filename
      filename = 'uns3d.msh'
      open(unit,file=filename,form='unformatted',status='old')
      rewind unit

      read (unit) ncell,nnodes,nedge,ncolort,ncolore,
     &            nnbound,nvbound,nfbound,
     &            nnfacet,nvfacet,nffacet,
     &            nsnode,nvnode,nfnode,ntte

      if (irank .eq. 0) write(10,100)ncell,nnodes,nedge,ncolort,ncolore
  100 format(1h ,'ncell nnodes nedge ncolort ncolore=',5(i8,1x))
      if (irank .eq. 0) write(10,110)nnbound,nvbound,nfbound
  110 format(1h ,'nnbound,nvbound,nfbound= ',3(i6,1x))
      if (irank .eq. 0) write(10,120)nnfacet,nvfacet,nffacet
  120 format(1h ,'nnfacet,nvfacet,nffacet= ',3(i6,1x))
      if (irank .eq. 0) write(10,130)nsnode,nvnode,nfnode,ntte
  130 format(1h ,'nsnode,nvnode,nfnode,ntte= ',4(i6,1x))

      return
      end


c================================ README =============================72
c
c Read grid 
c
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "README"
      subroutine README(nnodes,ncell,nedge,
     &                  ncolor,nccolor,
     &                  nnbound,nvbound,nfbound,
     &                  nnfacet,nvfacet,nffacet,
     &                  nsnode,nvnode,nfnode,ntte,
     &                  evec,x,y,z,vol,
     &                  c2n,c2e,
     &                  xn,yn,zn,ra,
     &                  nntet,nnpts,nvtet,nvpts,nftet,nfpts,
     &                  f2ntn,f2ntv,f2ntf,
     &                  isnode,sxn,syn,szn,
     &                  ivnode,vxn,vyn,vzn,
     &                  ifnode,fxn,fyn,fzn,slen,
     &                  irank)

      implicit none
#include "include/finclude/petscdef.h"
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      integer nnodes,ncell,nedge, 
     &        ncolor,nccolor,
     &        nnbound,nvbound,nfbound,
     &        nnfacet,nvfacet,nffacet,
     &        nsnode,nvnode,nfnode,ntte,
     &        irank
      integer unit
      integer evec(nedge,2)
      Scalar sxn(nsnode),syn(nsnode),szn(nsnode)
      Scalar vxn(nvnode),vyn(nvnode),vzn(nvnode)
      Scalar fxn(nfnode),fyn(nfnode),fzn(nfnode)
      Scalar vol(nnodes),slen(nnodes)
      Scalar x(nnodes),y(nnodes),z(nnodes)
      Scalar xn(nedge),yn(nedge),zn(nedge),ra(nedge)
c
c  Garbage variables
c
      integer c2n(ncell,4), c2e(ncell,6)
      integer nntet(nnbound), nnpts(nnbound)
      integer nvtet(nvbound), nvpts(nvbound)
      integer nftet(nfbound), nfpts(nfbound)
      integer f2ntn(nnfacet,4), f2ntv(nvfacet,4), f2ntf(nffacet,4)
      integer isnode(nsnode),ivnode(nvnode),ifnode(nfnode)
      integer idum,jdum,i,j,k

      unit = 20
c
c  Read Cell to Node indices
c
      if (ivisc.ne.0) then
         read (unit) (c2n(j,1),j=1,ncell),(c2n(j,2),j=1,ncell),
     &               (c2n(j,3),j=1,ncell),(c2n(j,4),j=1,ncell)
      else
         read (unit) (idum,j=1,ncell),(idum,j=1,ncell),
     &               (idum,j=1,ncell),(idum,j=1,ncell)
      end if

c
c  Read Cell to Edge indices
c
      if (ivisc.ne.0) then
         read (unit) (c2e(j,1),j=1,ncell),(c2e(j,2),j=1,ncell),
     &               (c2e(j,3),j=1,ncell),(c2e(j,4),j=1,ncell),
     &               (c2e(j,5),j=1,ncell),(c2e(j,6),j=1,ncell) 
      else
         read (unit) (idum,j=1,ncell),(idum,j=1,ncell),
     &               (idum,j=1,ncell),(idum,j=1,ncell),
     &               (idum,j=1,ncell),(idum,j=1,ncell) 
      end if
c
c  Read coordinates of grid points
c
      read (unit) (x(i),i=1,nnodes)
      read (unit) (y(i),i=1,nnodes)
      read (unit) (z(i),i=1,nnodes)
c
c  Read median dual volume surrounding each node
c
      read (unit) (vol(i),i=1,nnodes)
c
c  Read node indices for each edge
c
      read (unit) (evec(k,1),k=1,nedge),(evec(k,2),k=1,nedge)
c
c  Read in unit normals of dual mesh face (from node 1 to node 2)
c  and the area of the dual mesh face 
c
      read (unit) (xn(k),k=1,nedge),(yn(k),k=1,nedge),
     &            (zn(k),k=1,nedge),(ra(k),k=1,nedge)
c
c  Read Cell colors
c
      read (unit) (idum,i=1,nccolor)
c
c  Read edge colors
c
      read (unit) (idum,i=1,ncolor)
c
c  Read solid boundary face colors
c
      do i=1, nnbound
         read (unit) idum,(jdum,j=1,idum)
      end do
c
c  Read viscous boundary face colors
c
      do i=1, nvbound
         read (unit) idum,(jdum,j=1,idum)
      end do
c
c  Read boundary face colors
c
      do i=1, nfbound
         read (unit) idum,(jdum,j=1,idum)
      end do
c
c  Read inviscid surface info
c
      read (unit) (nntet(i),i=1,nnbound)
      read (unit) (nnpts(i),i=1,nnbound)
c
c  Read viscous surface info
c
      read (unit) (nvtet(i),i=1,nvbound)
      read (unit) (nvpts(i),i=1,nvbound)
c
c  Read far field info
c
      read (unit) (nftet(i),i=1,nfbound)
      read (unit) (nfpts(i),i=1,nfbound)
c
c  Read list of faces on inviscid boundaries
c
      read (unit) (f2ntn(i,1),i=1,nnfacet),(f2ntn(i,2),i=1,nnfacet),
     &            (f2ntn(i,3),i=1,nnfacet),(f2ntn(i,4),i=1,nnfacet)
c
c  Read list of faces on viscous boundaries
c
      read (unit) (f2ntv(i,1),i=1,nvfacet),(f2ntv(i,2),i=1,nvfacet),
     &            (f2ntv(i,3),i=1,nvfacet),(f2ntv(i,4),i=1,nvfacet)
c
c  Read list of faces on far field boundaries
c
      read (unit) (f2ntf(i,1),i=1,nffacet),(f2ntf(i,2),i=1,nffacet),
     &            (f2ntf(i,3),i=1,nffacet),(f2ntf(i,4),i=1,nffacet)
c
c  Read list of info associated with nodes on each inviscid boundary
c
      read (unit) (isnode(i),i=1,nsnode)
      read (unit) (sxn(i),i=1,nsnode),(syn(i),i=1,nsnode),
     &            (szn(i),i=1,nsnode)
c
c  Read list of info associated with nodes on each viscous boundary
c
      read (unit) (ivnode(i),i=1,nvnode)
      read (unit) (vxn(i),i=1,nvnode),(vyn(i),i=1,nvnode),
     &            (vzn(i),i=1,nvnode)
c
c  Read list of info associated with nodes on each far field boundary
c
      read (unit) (ifnode(i),i=1,nfnode)
      read (unit) (fxn(i),i=1,nfnode),(fyn(i),i=1,nfnode),
     &            (fzn(i),i=1,nfnode)
c
c Read in the distance function
c
      if (ivisc.gt.3) then
         read(unit)(slen(i),i=1,nnodes)
      end if
 
      close (unit)
c
c  End of subroutine README
c
      return
      end


c================================ PLLAN ====================================
c
c  Writes output for plotting 
c
c============================================================================
#undef __FUNC__
#define __FUNC__ "PLLAN"
      subroutine PLLAN(nnodes,irank)
      implicit none
#include "include/finclude/petscdef.h"
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      Scalar rms(1001),clw(1001),cdw(1001),
     &       cmw(1001),xres(1001)
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/history/rms,clw,cdw,cmw,xres
      integer nnodes,irank,ncolumn,n
      Scalar tpnpi
c
c Write out the residual history
c
c     tpnpi = (xres(ntt) - xres(1))/float(ntt*nnodes)
      tpnpi = ntt*nnodes
      tpnpi = xres(ntt)/tpnpi
      write(12,300) tpnpi
      ncolumn = 6
      write(12,100) ncolumn
      write(12,301)
      write(12,302)
      write(12,303)
      write(12,304)
      write(12,305)
      write(12,306)
      write(12,100) ntt
      do 1200 n = 1,ntt
c        xres(n) = xres(n)/1.0e6
         write(12,100)n,rms(n),clw(n),cdw(n),cmw(n),xres(n)
 1200 continue
  100 format(1h ,i5,1x,e14.7,2x,e14.7,2x,e14.7,2x,e14.7,2x,e14.7)
  300 format(1h ,'#time/node/iteration:',e14.7)
  301 format(1h ,'   v1::Iteration')
  302 format(1h ,'   v2::Log(R)')
  303 format(1h ,'   v3::c_l')
  304 format(1h ,'   v4::c_d')
  305 format(1h ,'   v5::c_m')
  306 format(1h ,'   v6::CPU time')
c
c End of subroutine PLLAN
c
      return
      end


c================================ TECFLO =============================72
c
c  Writes a formatted file that contains an input file for TECPLOT
c
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "TECFLO"
      subroutine TECFLO(nnodes,
     &                  nnbound,nvbound,nfbound,
     3                  nnfacet,nvfacet,nffacet,
     &                  nsnode,nvnode,nfnode,
     4                  title,x,y,z,q,
     5                  nnpts,nntet,nvpts,nvtet,nfpts,nftet,
     6                  f2ntn,f2ntv,f2ntf,isnode,ivnode,ifnode,
     7                  irank)
c
      implicit none      
#include "include/finclude/petscdef.h"
      Scalar pinf
      Scalar gamma,gm1,gp1,gm1g,gp1g,ggm1
      Scalar p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      integer nnodes,nnbound,nvbound,nfbound,
     &        nnfacet,nvfacet,nffacet,
     &        nsnode,nvnode,nfnode,irank
      integer nntet(1),nnpts(1)
      integer nvtet(1),nvpts(1)
      integer nftet(1),nfpts(1)
      integer f2ntn(nnfacet,4),f2ntv(nvfacet,4),f2ntf(nffacet,4)
      integer isnode(1),ivnode(1),ifnode(1)
      Scalar  x(1),y(1),z(1),q(4,nnodes)
      integer i,j,i1,i2,i3,ic,isp,ist
 
 
      character c4*4,title*20

      pinf = p0
c
c   + write tecplot header
c
      write(13,'(a)') 'TITLE="' // title // '"'
      write(13,'(a)') 'VARIABLES="X     ","Y     ","Z     ","RHO   ",    
     1     "U     ","V     ","W     ","P/Pinf"'  
c
c   + do a zone-title, so we can keep track of things
c   + start with solid-wall boundary surfaces
c  
      isp   = 1
      ist   = 1
      do 10 i=1,nnbound
        write(c4,"(i4)") i
        if (i .ge.    0 .and. i .le.    9) ic = 4
        if (i .ge.   10 .and. i .le.   99) ic = 3
        if (i .ge.  100 .and. i .le.  999) ic = 2
        if (i .ge. 1000 .and. i .le. 9999) ic = 1
 
        write(13,1000) "nn." // c4(ic:4),nnpts(i),nntet(i)
 
        write(13,1010) (x(isnode(j))      ,j=isp,isp+nnpts(i)-1)
        write(13,1010) (y(isnode(j))      ,j=isp,isp+nnpts(i)-1)
        write(13,1010) (z(isnode(j))      ,j=isp,isp+nnpts(i)-1)
        write(13,1010) (1.0                 ,j=isp,isp+nnpts(i)-1)
        write(13,1010) (q(2,isnode(j))      ,j=isp,isp+nnpts(i)-1)
        write(13,1010) (q(3,isnode(j))      ,j=isp,isp+nnpts(i)-1)
        write(13,1010) (q(4,isnode(j))      ,j=isp,isp+nnpts(i)-1)           
        write(13,1010) (q(1,isnode(j))/pinf ,j=isp,isp+nnpts(i)-1)
 
        do 30 j=ist,ist+nntet(i)-1
          i1 = f2ntn(j,1) - isp + 1
          i2 = f2ntn(j,2) - isp + 1
          i3 = f2ntn(j,3) - isp + 1
 
          write(13,1020) i1,i2,i3,i3
   30   continue
 
        isp = isp + nnpts(i) 
        ist = ist + nntet(i)
 
   10 continue
c
c   + start with viscous boundary surfaces
c 
      isp   = 1
      ist   = 1
      do 70 i=1,nvbound
        write(c4,"(i4)") i
        if (i .ge.    0 .and. i .le.    9) ic = 4
        if (i .ge.   10 .and. i .le.   99) ic = 3
        if (i .ge.  100 .and. i .le.  999) ic = 2
        if (i .ge. 1000 .and. i .le. 9999) ic = 1
 
        write(13,1000) "nv." // c4(ic:4),nvpts(i),nvtet(i)
 
        write(13,1010) (x(ivnode(j))      ,j=isp,isp+nvpts(i)-1)
        write(13,1010) (y(ivnode(j))      ,j=isp,isp+nvpts(i)-1)
        write(13,1010) (z(ivnode(j))      ,j=isp,isp+nvpts(i)-1)
        write(13,1010) (1.0                 ,j=isp,isp+nvpts(i)-1)
        write(13,1010) (q(2,ivnode(j))      ,j=isp,isp+nvpts(i)-1)
        write(13,1010) (q(3,ivnode(j))      ,j=isp,isp+nvpts(i)-1)
        write(13,1010) (q(4,ivnode(j))      ,j=isp,isp+nvpts(i)-1)           
        write(13,1010) (q(1,ivnode(j))/pinf ,j=isp,isp+nvpts(i)-1)
 
        do 90 j=ist,ist+nvtet(i)-1
           i1 = f2ntv(j,1) - isp + 1
           i2 = f2ntv(j,2) - isp + 1
           i3 = f2ntv(j,3) - isp + 1
 
           write(13,1020) i1,i2,i3,i3
   90   continue
 
        isp = isp + nvpts(i)
        ist = ist + nvtet(i)
 
   70 continue
c
c   + do the far-field boundary surfaces
c  
      isp   = 1
      ist   = 1
      do 40 i=1,nfbound
         write(c4,"(i4)") i
         if (i .ge.    0 .and. i .le.    9) ic = 4
         if (i .ge.   10 .and. i .le.   99) ic = 3
         if (i .ge.  100 .and. i .le.  999) ic = 2
         if (i .ge. 1000 .and. i .le. 9999) ic = 1
  
         write(13,1000) "ff." // c4(ic:4),nfpts(i),nftet(i)
  
         write(13,1010) (x(ifnode(j))        ,j=isp,isp+nfpts(i)-1)         
         write(13,1010) (y(ifnode(j))        ,j=isp,isp+nfpts(i)-1)
         write(13,1010) (z(ifnode(j))        ,j=isp,isp+nfpts(i)-1)
         write(13,1010) (1.0                 ,j=isp,isp+nfpts(i)-1)
         write(13,1010) (q(2,ifnode(j))      ,j=isp,isp+nfpts(i)-1)
         write(13,1010) (q(3,ifnode(j))      ,j=isp,isp+nfpts(i)-1)
         write(13,1010) (q(4,ifnode(j))      ,j=isp,isp+nfpts(i)-1)
         write(13,1010) (q(1,ifnode(j))/pinf ,j=isp,isp+nfpts(i)-1)
 
         do 60 j=ist,ist+nftet(i)-1
            i1 = f2ntf(j,1) - isp + 1
            i2 = f2ntf(j,2) - isp + 1
            i3 = f2ntf(j,3) - isp + 1
 
            write(13,1020) i1,i2,i3,i3
   60    continue
 
         isp = isp + nfpts(i) 
         ist = ist + nftet(i)
 
   40  continue

c
c      End of subroutine TECFLO
c
 1000  format('ZONE T="',a,'", Z=0, I=',i6,', J=',i6,', F=FEBLOCK')
 1010  format(1P10E13.5)
 1020  format(4I10)

       return
       end


c=============================== FASFLO ==============================72
c
c Write a FAST binary file
c
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "FASFLO"
      subroutine FASFLO(nnodes,nsnode,nnfacet,isnode,f2ntn,x,y,z,qvec)

      implicit none      
#include "include/finclude/petsc.h"
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      integer nnodes,nsnode,nnfacet
      integer isnode(1),f2ntn(nnfacet,1)
      Scalar qvec(4,nnodes)
      Scalar x(1),y(1),z(1)
      Scalar time
      integer i,j,nfaces,nx
c
      nfaces=nnfacet
      time=1.0d0

      write (14) nsnode,nfaces,0
      write (14) (x(isnode(i)),i=1,nsnode),
     &           (y(isnode(i)),i=1,nsnode),
     &           (z(isnode(i)),i=1,nsnode),
     &           ((f2ntn(i,j),j=1,3),i=1,nnfacet),
     &           (1,i=1,nnfacet)

      write (15) nsnode,1,1
      write (15) beta,alpha,Re,time
      write (15) (1.0,i=1,nsnode),
     &           ((qvec(nx,isnode(i)),nx=2,4),i=1,nsnode),
     &            (qvec(isnode(i),1),i=1,nsnode)

c
c End of subroutine FASFLO
c
      return
      end


c================================ L2NORM =============================72
c
c  calculates the L2 norm of the residual
c  I have eliminated dq - D. K. Kaushik (1/23/97)
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "L2NORM"
      subroutine L2NORM(res,nnodes,nnode_glo,x,y,z,vol,irank)
      implicit none
#include "include/finclude/petsc.h"  
      integer nnodes,nnode_glo,irank,icount
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      Scalar  rms(1001),clw(1001),cdw(1001),
     &        cmw(1001),xres(1001)
      Scalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,
     &        nstage,ncyct,iramp,nitfo
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/history/rms,clw,cdw,cmw,xres
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo

      Scalar res(4,nnodes),x(1),y(1),z(1)
c     dimension dq(nnodes,4),vol(1)
      Scalar vol(1)
      integer iflag,i,ierr
      Scalar sum,sumglo,t1,rmax,test,xloc,yloc,zloc,
     &       ratio,compare

c
c Set the flag to monitor which type of residual you want
c iflag=0 "normal" residual
c iflag=1 residual/volume
c iflag=2 dq
c
      iflag = 0
 
      t1 = float(nnode_glo)
      sum = 0.0d0
      rmax = 0.0d0
c
c iflag=0
c
      if (iflag.eq.0) then
         do 1010 i = 1,nnodes
            sum = sum + res(1,i)*res(1,i)
            test = abs(res(1,i))
            if (test.ge.rmax)then
               rmax = test
               xloc = x(i)
               yloc = y(i)
               zloc = z(i)
            end if
 1010    continue
      end if
c
c iflag=1
c
      if (iflag.eq.1) then
         do 1020 i = 1,nnodes
            sum = sum + res(1,i)*res(1,i)/vol(i)/vol(i)
            test = abs(res(1,i)/vol(i))
            if (test.ge.rmax)then
               rmax = test
               xloc = x(i)
               yloc = y(i)
               zloc = z(i)
            end if
 1020    continue
      end if
c
c iflag=2
c
c     if (iflag.eq.2)then
c        do 1030 i = 1,nnodes
c           sum = sum + dq(i,1)*dq(i,1)
c           test = abs(dq(i,1))
c           if (test.ge.rmax)then
c              rmax = test
c              xloc = x(i)
c              yloc = y(i)
c              zloc = z(i)
c           end if
c1030    continue
c     end if
 
c     rms(ntt) = sqrt(sum/t1)
      icount = 1
      sumglo=0.0
c     write(*,*) 'I am in L2NORM just before Reduce op'
c     write(*,*) 'Local res. norm on processor ',irank,' is ',sqrt(sum)
      call MPI_ALLREDUCE(sum,sumglo,icount,
     >                   MPIU_SCALAR, MPI_SUM,
     >                   MPI_COMM_WORLD,ierr)
c     write(*,*) 'In L2NORM - residual norm is', sqrt(sumglo)
      rms(ntt) = sqrt(sumglo/t1)
c
c If we are ramping the cfl with SER save res0 and resc
c
      if (ntt.eq.1) res0 = rms(ntt)
      resc = rms(ntt)
      compare = res0*cfl1/cfl2
      if (resc.lt.compare) resc = compare
      ratio = res0/resc

c     write(6,100)ntt,rms(ntt),rmax,xloc,yloc,zloc,ratio
      if (irank .eq. 0) write(10,100)ntt,rms(ntt),rmax,
     >                 xloc,yloc,zloc,ratio

  100 format(1h ,'Iteration',i6,' rms rmax xloc yloc zloc rat = ',
     &       6(1e14.7,1x))
  202 format(1h ,2x,e14.7,2x,e14.7,2x,i6)
c
c End of subroutine L2NORM
c
      return
      end


c================================ FORCE  =============================72
c
c  Calculates the forces
c  Modified - D. K. Kaushik (1/15/97)
c  Added new parameters - clift, cdrag, cmom, irank, nvertices
c  
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "FORCE"
      subroutine FORCE(nnodes,nedge,
     &                 isnode,ivnode,
     &                 nnfacet,f2ntn,nnbound,
     &                 nvfacet,f2ntv,nvbound,
     &                 evec,qvec,xyz,sface_bit,vface_bit,
     &                 clift, cdrag, cmom,irank,nvertices)
      implicit none      
#include "include/finclude/petsc.h"
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      Scalar gamma,gm1,gp1,gm1g,gp1g,ggm1
      Scalar p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      Scalar rms(1001),clw(1001),cdw(1001),
     &       cmw(1001),xres(1001)
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/history/rms,clw,cdw,cmw,xres

      integer nnodes,nedge,nnfacet,nvfacet,
     &        nnbound,nvbound,
     &        irank,nvertices
      integer isnode(1),ivnode(1)
      integer f2ntn(nnfacet,4)
      integer f2ntv(nvfacet,4)
      integer sface_bit(nnfacet), vface_bit(nvfacet)

      Scalar clift, cdrag, cmom
      Scalar clloc,cdloc,cmloc,clglo,cdglo,cmglo
      Scalar cl_ghost, cd_ghost, cm_ghost
      Scalar coef_loc(3), coef_glo(3)
      Scalar pi,conv,csa,sna,xr,yr,zr,
     &       x1,y1,z1,x2,y2,z2,x3,y3,z3,xmid,ymid,zmid,
     &       p1,p2,p3,u1,u2,u3,v1,v2,v3,
     &       w1,w2,w3,
     &       ax,ay,az,bx,by,bz,
     &       xnorm,ynorm,znorm,
     &       dcx,dcy,dcz,cp,press
      integer icount,n,node1,node2,node3,ierr
      
#if defined(INTERLACING)
      Scalar qvec(4,nvertices)
      Scalar xyz(3,nvertices)
      integer evec(2,nedge)
#define qnode(i,j) qvec(i,j)
#define x(i) xyz(1,i)
#define y(i) xyz(2,i)
#define z(i) xyz(3,i)
#define eptr(j,i) evec(i,j)
#else
      Scalar qvec(nvertices,4)
      Scalar xyz(nvertices,3)
      integer evec(nedge,2)
#define qnode(i,j) qvec(j,i)
#define x(i) xyz(i,1)
#define y(i) xyz(i,2)
#define z(i) xyz(i,3)
#define eptr(i,j) evec(i,j)
#endif
c
      pi = 4.d0*atan(1.d0)
      conv = 180.d0/pi
      csa=cos(alpha/conv)
      sna=sin(alpha/conv)
c
c  initialize forces to zero  
c
      clw(ntt) = 0.0d0
      cdw(ntt) = 0.0d0
      cmw(ntt) = 0.0d0
      cl_ghost = 0.0d0
      cd_ghost = 0.0d0
      cm_ghost = 0.0d0
      clloc = 0.0d0
      cdloc = 0.0d0
      cmloc = 0.0d0
      clglo = 0.0d0
      cdglo = 0.0d0
      cmglo = 0.0d0
      coef_glo(1) = 0.0d0
      coef_glo(2) = 0.0d0
      coef_glo(3) = 0.0d0
 
      xr = 0.25d0
      yr = 0.0d0
      zr = 0.0d0
       
      do 30 n = 1, nnfacet
               node1 = isnode(f2ntn(n,1))
               node2 = isnode(f2ntn(n,2))
               node3 = isnode(f2ntn(n,3))

               x1    = x(node1)
               y1    = y(node1)
               z1    = z(node1)

               x2    = x(node2)
               y2    = y(node2)
               z2    = z(node2)

               x3 = x(node3)
               y3 = y(node3)
               z3 = z(node3)
               
               ax = x2 - x1
               ay = y2 - y1
               az = z2 - z1

               bx = x3 - x1
               by = y3 - y1
               bz = z3 - z1
c
c  norm points outward, away from grid interior.
c  norm magnitude is area of surface triangle.
c
               xnorm =-0.5d0*(ay*bz - az*by)
               ynorm = 0.5d0*(ax*bz - az*bx)
               znorm =-0.5d0*(ax*by - ay*bx)
 
               p1 = qnode(1,node1)
               u1 = qnode(2,node1)
               v1 = qnode(3,node1)
               w1 = qnode(4,node1)

               p2 = qnode(1,node2)
               u2 = qnode(2,node2)
               v2 = qnode(3,node2)
               w2 = qnode(4,node2)

               p3 = qnode(1,node3)
               u3 = qnode(2,node3)
               v3 = qnode(3,node3)
               w3 = qnode(4,node3)
 
               press = (p1 + p2 + p3)/3.0d0
               cp    = 2.d0*(press - 1.0d0)        
c
               dcx = cp*xnorm
               dcy = cp*ynorm
               dcz = cp*znorm

               xmid = (x1 + x2 + x3)
               ymid = (y1 + y2 + y3)
               zmid = (z1 + z2 + z3)

               if (sface_bit(n) .eq. 1) then
#if defined(CFL3D_AXIS)

                clloc = clloc - dcx*sna     + dcz*csa
                cdloc = cdloc + dcx*csa + dcz*sna
                cmloc = cmloc
     &                     + (xmid - xr)*dcy - (ymid - yr)*dcx

#else
                clloc = clloc - dcx*sna + dcy*csa
                cdloc = cdloc + dcx*csa + dcy*sna
                cmloc = cmloc
     &                     + (xmid - xr)*dcy - (ymid - yr)*dcx
#endif
               endif

 30    continue
c
c Viscous boundary
c
       do 60 n = 1, nvfacet
               node1 = ivnode(f2ntv(n,1))
               node2 = ivnode(f2ntv(n,2))
               node3 = ivnode(f2ntv(n,3))

               x1    = x(node1)
               y1    = y(node1)
               z1    = z(node1)

               x2    = x(node2)
               y2    = y(node2)
               z2    = z(node2)

               x3 = x(node3)
               y3 = y(node3)
               z3 = z(node3)
               
               ax = x2 - x1
               ay = y2 - y1
               az = z2 - z1

               bx = x3 - x1
               by = y3 - y1
               bz = z3 - z1
c
c  norm points outward, away from grid interior.
c  norm magnitude is area of surface triangle.
c
               xnorm =-0.5d0*(ay*bz - az*by)
               ynorm = 0.5d0*(ax*bz - az*bx)
               znorm =-0.5d0*(ax*by - ay*bx)
 
               p1    = qnode(1,node1)
               u1    = qnode(2,node1)
               v1    = qnode(3,node1)
               w1    = qnode(4,node1)

               p2    = qnode(1,node2)
               u2    = qnode(2,node2)
               v2    = qnode(3,node2)
               w2    = qnode(4,node2)

               p3    = qnode(1,node3)
               u3    = qnode(2,node3)
               v3    = qnode(3,node3)
               w3    = qnode(4,node3)
 
               press = (p1 + p2 + p3)/3.0d0
               cp    = 2.d0*(press -1.0d0)
 
               dcx = cp*xnorm
               dcy = cp*ynorm
               dcz = cp*znorm

               xmid = (x1 + x2 + x3)
               ymid = (y1 + y2 + y3)
               zmid = (z1 + z2 + z3)

               if (vface_bit(n) .eq. 1) then
                clloc = clloc - dcx*sna + dcy*csa
                cdloc = cdloc + dcx*csa + dcy*sna
                cmloc = cmloc
     &                     + (xmid - xr)*dcy - (ymid - yr)*dcx
               endif

 60   continue
c
      icount = 3
      coef_loc(1) = clloc
      coef_loc(2) = cdloc
      coef_loc(3) = cmloc
      call MPI_ALLREDUCE(coef_loc,coef_glo,icount,
     >                   MPIU_SCALAR, MPI_SUM,
     >                   MPI_COMM_WORLD,ierr)
c     call MPI_ALLREDUCE(cdloc,cdglo,icount,
c    >                   MPI_DOUBLE_PRECISION, MPI_SUM,
c    >                   MPI_COMM_WORLD,ierr)
c     call MPI_ALLREDUCE(cmloc,cmglo,icount,
c    >                   MPI_DOUBLE_PRECISION, MPI_SUM,
c    >                   MPI_COMM_WORLD,ierr)
c
       clw(ntt) = coef_glo(1)
       cdw(ntt) = coef_glo(2)
       cmw(ntt) = coef_glo(3)
       clift = clw(ntt)
       cdrag = cdw(ntt)
       cmom = cmw(ntt)
c
c Update the timing information - Added by D. K. Kaushik (1/17/97)
      xres(ntt) = tot
c End of subroutine FORCE
c
      return
      end


c================================ DELTAT2 ============================72
c
c Calculate a time step for each cell
c Note that this routine assumes conservative variables
c
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "DELTA2"
      subroutine DELTAT2(nnodes,nedge,qvec,cdt,
     &                  xyz,vol,xyzn,evec,
     &                  sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,
     &                  nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
     &                  irank,nvertices)
c
      implicit none      
#include "include/finclude/petscdef.h"
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      Scalar  gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      integer nnodes,nedge,
     &        nsnode,nvnode,nfnode,
     &        irank,nvertices
      Scalar  vol(1)
      Scalar  sxn(1),syn(1),szn(1)
      Scalar  vxn(1),vyn(1),vzn(1)
      Scalar  fxn(1),fyn(1),fzn(1)
      Scalar  cdt(1)
      integer isnode(1),ivnode(1),ifnode(1)
      integer n,i,node1,node2,inode
      integer flops
      Scalar xnorm,ynorm,znorm,area,
     &       u1,v1,w1,u2,v2,w2,u,v,c,w,
     &       ubar,Vn,term
#if defined(INTERLACING)
      Scalar qvec(4,nvertices)
      Scalar xyz(3,nvertices)
      Scalar xyzn(4,nedge)
      integer evec(2,nedge)
#define qnode(i,j) qvec(i,j)
#define x(i) xyz(1,i)
#define y(i) xyz(2,i)
#define z(i) xyz(3,i)
#define xn(i) xyzn(1,i)
#define yn(i) xyzn(2,i)
#define zn(i) xyzn(3,i)
#define rl(i) xyzn(4,i)
#define eptr(j,i) evec(i,j)
#else
      Scalar qvec(nvertices,4)
      Scalar xyz(nvertices,3)
      Scalar xyzn(nedge,4)
      integer evec(nedge,2)
#define qnode(i,j) qvec(j,i)
#define x(i) xyz(i,1)
#define y(i) xyz(i,2)
#define z(i) xyz(i,3)
#define xn(i) xyzn(i,1)
#define yn(i) xyzn(i,2)
#define zn(i) xyzn(i,3)
#define rl(i) xyzn(i,4)
#define eptr(i,j) evec(i,j)
#endif

c  If local time steping, loop over faces
c  and calculate time step as cdt = V/(sum(|u.n| +c.area)
c  This is time step for cfl=1. We will multiply by cfl number later
c
c First loop over nodes and zero out cdt
c
      flops = 0
      if (dt.lt.0.0) then
         do 1000 i = 1,nvertices
            cdt(i) = 0.0d0
 1000    continue
c
         do 1020 n = 1, nedge
               node1 = eptr(n,1)
               node2 = eptr(n,2)
c
c Get normal to face
c
               xnorm = xn(n)
               ynorm = yn(n)
               znorm = zn(n)
               area  = rl(n)
c
               xnorm = xnorm*area
               ynorm = ynorm*area
               znorm = znorm*area
c /*'
c xnorm = x-normal x area of face  
c ynorm = y-normal x area of face  
c znorm = z-normal x area of face  
c '*/
               u1   = qnode(2,node1)
               v1   = qnode(3,node1)
               w1   = qnode(4,node1)
 
               u2   = qnode(2,node2)
               v2   = qnode(3,node2)
               w2   = qnode(4,node2)
c
c Get average values on face 
c
               u    = 0.5d0*(u1 + u2)
               v    = 0.5d0*(v1 + v2)
               w    = 0.5d0*(w1 + w2)
               ubar = xn(n)*u + yn(n)*v + zn(n)*w
               c    = sqrt(ubar*ubar + beta)
 
               term = abs(u*xnorm + v*ynorm + w*znorm) + c*area
               cdt(node1) = cdt(node1) + term
               cdt(node2) = cdt(node2) + term
c
 1020    continue
c        call PLogFlops(27*nedge)
         flops = flops + 27*nedge
c 
c Now loop over boundaries and close the contours
c
         do 1030 i = 1,nsnode
            inode = isnode(i)
c
c Get the normal
c
            xnorm = sxn(i)
            ynorm = syn(i)
            znorm = szn(i)
            area  = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
 
            u   = qnode(2,inode)
            v   = qnode(3,inode)
            w   = qnode(4,inode)
            ubar= (u*xnorm + v*ynorm + w*znorm)/area
            c   = sqrt(ubar*ubar + beta)
c
            Vn = abs(xnorm*u + ynorm*v + znorm*w) + c*area
            cdt(inode) = cdt(inode) + Vn
c
 1030    continue
c        call PLogFlops(24*nsnode)
         flops = flops + 24*nsnode

c
c Now viscous faces
c
c
         do 1040 i = 1,nvnode
            inode = ivnode(i)
c
c Get the normal
c
            xnorm = vxn(i)
            ynorm = vyn(i)
            znorm = vzn(i)
            area  = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
c
c Because velocities are zero, c is just sqrt(beta)
c
            c  = sqrt(beta)
            Vn = c*area 

            cdt(inode) = cdt(inode) + Vn
c
 1040    continue
c        call PLogFlops(9*nvnode)
         flops = flops + 9*nvnode
c
c Now far field
c
         do 1050 i = 1,nfnode
            inode = ifnode(i)
c
c Get the normal
c
            xnorm = fxn(i)
            ynorm = fyn(i)
            znorm = fzn(i)
            area  = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
 
            u   = qnode(2,inode)
            v   = qnode(3,inode)
            w   = qnode(4,inode)
            ubar= (u*xnorm + v*ynorm + w*znorm)/area
            c = sqrt(ubar*ubar + beta)
 
            Vn = abs(xnorm*u + ynorm*v + znorm*w) + c*area
            cdt(inode) = cdt(inode) + Vn
 1050    continue
c        call PLogFlops(24*nfnode)
         flops = flops + 24*nfnode
 
         do 1060 n = 1,nvertices
            cdt(n) = vol(n)/cdt(n)
 1060    continue
c        call PLogFlops(nvertices)
         flops = flops + nvertices
      else
c
c If not doing local time stepping just set cdt=1
c
         do 1070 n = 1,nvertices
            cdt(n) = 1.0d0
 1070    continue
      end if
      call PLogFlops(flops)
c
c End of subroutine DELTAT2
c
      return
      end


c================================= FLUX ======================================
c
c Calculates the fluxes on the face and performs the flux balance
c
c=============================================================================
#undef __FUNC__
#define __FUNC__ "FLUX"
      subroutine FLUX(nnodes, ncell, nedge,
     &                nsface, nvface, nfface, isface, ivface, ifface,
     &                nsnode, nvnode, nfnode, isnode, ivnode, ifnode,
     &                nnfacet,f2ntn,nnbound,
     &                nvfacet,f2ntv,nvbound,
     &                nffacet,f2ntf,nfbound,
     &                grad,
     &                evec, qvec,
     &                xyz,
     &                resvec,resd,
     &                xyzn,
     &                sxn, syn, szn,
     &                vxn, vyn, vzn,
     &                fxn, fyn, fzn, phi, irank,nvertices)
      implicit none      
#include "include/finclude/petscdef.h"
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      Scalar  gamma,gm1,gp1,gm1g,gp1g,ggm1
      Scalar  p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      Scalar  rms(1001),clw(1001),cdw(1001),
     &        cmw(1001),xres(1001)
      Scalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,
     &        nstage,ncyct,iramp,nitfo
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/history/rms,clw,cdw,cmw,xres
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo
      integer nnodes, ncell, nedge,
     &        nsface, nvface, nfface,
     &        nsnode, nvnode, nfnode,
     &        nnfacet,nvfacet,nffacet,
     &        nnbound,nvbound,nfbound,
     &        irank,nvertices
      Scalar  sxn(1),syn(1),szn(1)
      Scalar  vxn(1),vyn(1),vzn(1)
      Scalar  fxn(1),fyn(1),fzn(1)
      Scalar  phi(nvertices,4)
      integer isface(1),ivface(1),ifface(1)
      integer isnode(1),ivnode(1),ifnode(1)
      integer f2ntn(nnfacet,4)
      integer f2ntv(nvfacet,4)
      integer f2ntf(nffacet,4)
      integer i,n,node1,node2,node3,inode      
      integer flops
      Scalar  xmean,ymean,zmean,xnorm,ynorm,znorm,
     &        rlen,area,dot,size,
     &        X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,p1,p2,p3,
     &        rx,ry,rz,pL,uL,wL,vL,pR,uR,vR,wR,
     &        ubarL,ubarR,ubar,p,u,v,w,c,c2,c20,
     &        ubar0,pi,ui,vi,wi,unorm,
     &        phi1,phi2,phi3,phi4,phi5,phi6,phi7,phi8,phi9,
     &        eig1,eig2,eig3,eig4,dp,du,dv,dw,
     &        ti11,ti21,ti31,ti41,dv1,dv2,dv3,dv4
      Scalar  r11,r21,r31,r41,r12,r22,r32,r42,
     &        r13,r23,r33,r43,r14,r24,r34,r44,
     &        t1,t2,t3,t4,
     &        t11,t21,t31,t41,t12,t22,t32,t42,
     &        t13,t23,t33,t43,t14,t24,t34,t44,
     &        fluxp1,fluxp2,fluxp3,fluxp4, 
     &        fluxm1,fluxm2,fluxm3,fluxm4
      Scalar  res1,res2,res3,res4,
     &        rhs1,rhs2,rhs3,rhs4,   
     &        c68,c18,second,
     &        ax,ay,az,bx,by,bz,
     &        pa,pb,pc,ub,vb,wb       
#if defined(INTERLACING)
      Scalar qvec(4,nvertices) 
      Scalar grad(3,4,nvertices) 
      Scalar resvec(4,nnodes)
      Scalar resd(4,nnodes)
      Scalar xyz(3,nvertices)
      Scalar xyzn(4,nedge)
      integer evec(2,nedge)
#define qnode(i,j) qvec(i,j)
#define res(i,j) resvec(i,j)
#define gradx(i,j) grad(1,i,j)
#define grady(i,j) grad(2,i,j)
#define gradz(i,j) grad(3,i,j)
#define x(i) xyz(1,i)
#define y(i) xyz(2,i)
#define z(i) xyz(3,i)
#define xn(i) xyzn(1,i)
#define yn(i) xyzn(2,i)
#define zn(i) xyzn(3,i)
#define ra(i) xyzn(4,i)
#define eptr(j,i) evec(i,j)
#else
      Scalar qvec(nvertices,4) 
      Scalar grad(nvertices,4,3) 
      Scalar resvec(nnodes,4)
      Scalar resd(nnodes,4)
      Scalar xyz(nvertices,3)
      Scalar xyzn(nedge,4)
      integer evec(nedge,2)
#define qnode(i,j) qvec(j,i)
#define res(i,j) resvec(j,i)
#define gradx(i,j) grad(j,i,1)
#define grady(i,j) grad(j,i,2)
#define gradz(i,j) grad(j,i,3)
#define x(i) xyz(i,1)
#define y(i) xyz(i,2)
#define z(i) xyz(i,3)
#define xn(i) xyzn(i,1)
#define yn(i) xyzn(i,2)
#define zn(i) xyzn(i,3)
#define ra(i) xyzn(i,4)
#define eptr(i,j) evec(i,j)
#endif
#if defined(_OPENMP)
       integer my_thread_id,omp_get_thread_num
#undef res(a,b)
#define res(a,b) resd(a,b)
       call PetscMemzero(resd,8*4*nnodes)
#endif

       flops = 0
       second = 1.0d0
c      if(ntt.le.nitfo)second = 0.0
c      print *, 'Second is' , second
c Loop over all the faces and calculate flux 
c
!$omp  parallel default(shared) private(n,node1,node2,xmean,ymean,
!$omp1 zmean,xnorm,ynorm,znorm,rlen,dot,x1,y1,z1,size,x2,y2,z2,rx,
!$omp2 ry,rz,pL,uL,vL,wL,ubarL,pR,uR,vR,wR,ubarR,p,u,v,w,ubar,phi1,
!$omp3 phi2,phi3,phi4,phi5,phi6,phi7,phi8,phi9,c2,c,eig1,eig2,eig3,
!$omp4 eig4,dp,du,dv,dw,ti11,ti21,ti31,ti41,dv1,dv2,dv3,dv4,r11,r21,
!$omp5 r31,r41,r12,r22,r32,r42,r13,r23,r33,r43,r14,r24,r34,r44,t1,t2,
!$omp6 t3,t4,fluxp1,fluxp2,fluxp3,fluxp4,fluxm1,fluxm2,fluxm3,fluxm4,
!$omp7 res1,res2,res3,res4) firstprivate(resd)
c!$omp8 private(my_thread_id)
c!$omp1 shared(evec,qvec,resvec,xyz,xyzn,grad,nedge,nnodes,second,flops,
c!$omp2 title,beta,alpha,Re,dt,tot,res0,resc,ntt,mseq,ivisc,irest,icyc,
c!$omp3 ihane,ntturb,gamma,gm1,gp1,gm1g,gp1g,ggm1,p0,rho0,c0,u0,v0,w0,
c!$omp4 et0,h0,pt0,rms,clw,cdw,cmw,xres,cfl1,cfl2,nsmoth,iflim,
c!$omp5 itran,nbtran,jupdate,nstage,ncyct,iramp,nitfo)
c!$      my_thread_id = omp_get_thread_num()
c!$omp   critical
c!$      print *, 'My thread id on processor ',irank,' is ',my_thread_id
c!$omp   end critical
!$omp   do reduction(+:flops)
       do n = 1, nedge
        node1 = eptr(n,1)
        node2 = eptr(n,2)
        if ((node1 .le. nnodes).or.(node2 .le. nnodes)) then
c
c Calculate unit normal to face and length of face
c
          xmean = .5d0*(x(node1) + x(node2))
          ymean = .5d0*(y(node1) + y(node2))
          zmean = .5d0*(z(node1) + z(node2))
          xnorm  = xn(n)
          ynorm  = yn(n)
          znorm  = zn(n)
          rlen   = ra(n)
c
c Now lets get our other 2 vectors
c For first vector, use {1,0,0} and subtract off the component
c in the direction of the face normal. If the inner product of
c {1,0,0} is close to unity, use {0,1,0}
c
         dot = xnorm
         if(abs(dot).lt.0.95d0)then
          X1 = 1. - dot*xnorm
          Y1 =    - dot*ynorm
          Z1 =    - dot*znorm
         else
          dot = ynorm
          X1 =    - dot*xnorm
          Y1 = 1.0d0 - dot*ynorm
          Z1 =    - dot*znorm
         end if
c
c Normalize the first vector
c
         size = sqrt(X1*X1 + Y1*Y1 + Z1*Z1)
         X1 = X1/size
         Y1 = Y1/size
         Z1 = Z1/size
c
c Take cross-product of normal and V1 to get V2
c
         X2 = ynorm*Z1 - znorm*Y1
         Y2 = znorm*X1 - xnorm*Z1
         Z2 = xnorm*Y1 - ynorm*X1
c
c Get variables on "left" and "right" side of face
c
         rx     = second*(xmean - x(node1))
         ry     = second*(ymean - y(node1))
         rz     = second*(zmean - z(node1))
         pL = qnode(1,node1) + gradx(1,node1)*rx 
     1                       + grady(1,node1)*ry
     1                       + gradz(1,node1)*rz
         uL = qnode(2,node1) + gradx(2,node1)*rx 
     1                       + grady(2,node1)*ry 
     1                       + gradz(2,node1)*rz 
         vL = qnode(3,node1) + gradx(3,node1)*rx 
     1                       + grady(3,node1)*ry 
     1                       + gradz(3,node1)*rz 
         wL = qnode(4,node1) + gradx(4,node1)*rx 
     1                       + grady(4,node1)*ry 
     1                       + gradz(4,node1)*rz 

         ubarL  = xnorm*uL + ynorm*vL + znorm*wL
c        c2L = ubarl*ubarL + beta
c        cL  = sqrt(c2L)
c
         rx     = second*(xmean - x(node2))
         ry     = second*(ymean - y(node2))
         rz     = second*(zmean - z(node2))
         pR   = qnode(1,node2) + gradx(1,node2)*rx 
     1                         + grady(1,node2)*ry
     1                         + gradz(1,node2)*rz
         uR   = qnode(2,node2) + gradx(2,node2)*rx 
     1                         + grady(2,node2)*ry 
     1                         + gradz(2,node2)*rz 
         vR   = qnode(3,node2) + gradx(3,node2)*rx 
     1                         + grady(3,node2)*ry 
     1                         + gradz(3,node2)*rz 
         wR   = qnode(4,node2) + gradx(4,node2)*rx 
     1                         + grady(4,node2)*ry 
     1                         + gradz(4,node2)*rz 
         ubarR  = xnorm*uR + ynorm*vR + znorm*wR
c        c2R = ubarR*ubarR + beta
c        cR  = sqrt(c2R)
c
c Compute averages
c
          p = .5d0*(pL + pR)
          u = .5d0*(uL + uR)
          v = .5d0*(vL + vR)
          w = .5d0*(wL + wR)

          ubar  = xnorm*u + ynorm*v + znorm*w
          phi1  = xnorm*beta + u*ubar
          phi2  = ynorm*beta + v*ubar
          phi3  = znorm*beta + w*ubar
          phi4  = Y2*phi3 - Z2*phi2
          phi5  = Z2*phi1 - X2*phi3
          phi6  = X2*phi2 - Y2*phi1
          phi7  = Z1*phi2 - Y1*phi3
          phi8  = X1*phi3 - Z1*phi1
          phi9  = Y1*phi1 - X1*phi2
          c2    = ubar*ubar + beta
          c     = sqrt(c2)
c
c Now compute eigenvalues, eigenvectors, and strengths
c
          eig1 = abs(ubar)
          eig2 = abs(ubar)
          eig3 = abs(ubar + c)
          eig4 = abs(ubar - c)
c
          dp = pr - pl
          du = ur - ul
          dv = vr - vl
          dw = wr - wl
c
c Components of T(inverse) (I will divide by c2 later)
c
          ti11 = -(u*phi4 + v*phi5 + w*phi6)/beta
          ti21 = -(u*phi7 + v*phi8 + w*phi9)/beta
          ti31 =  .5d0*(c-ubar)/beta
          ti41 = -.5d0*(c+ubar)/beta
c jumps (T(inverse)*dq) 
          dv1 = (ti11*dp + phi4*du + phi5*dv + phi6*dw)/c2
          dv2 = (ti21*dp + phi7*du + phi8*dv + phi9*dw)/c2
          dv3 = .5d0*(2.d0*ti31*dp + xnorm*du + ynorm*dv + znorm*dw)/c2
          dv4 = .5d0*(2.d0*ti41*dp + xnorm*du + ynorm*dv + znorm*dw)/c2
c
c Now get elements of T (call it r for now)
c
          r11 = 0.0d0
          r21 = X1
          r31 = Y1
          r41 = Z1

          r12 = 0.0d0
          r22 = X2
          r32 = Y2
          r42 = Z2

          r13 = c*beta
          r23 = xnorm*beta + u*(ubar + c)
          r33 = ynorm*beta + v*(ubar + c)
          r43 = znorm*beta + w*(ubar + c)
c
          r14 = -c*beta
          r24 = xnorm*beta + u*(ubar - c)
          r34 = ynorm*beta + v*(ubar - c)
          r44 = znorm*beta + w*(ubar - c)
c
c Calculate T* |lambda| *T(inverse)
c
          t1 = eig1*r11*dv1 + eig2*r12*dv2 + eig3*r13*dv3 + eig4*r14*dv4
          t2 = eig1*r21*dv1 + eig2*r22*dv2 + eig3*r23*dv3 + eig4*r24*dv4
          t3 = eig1*r31*dv1 + eig2*r32*dv2 + eig3*r33*dv3 + eig4*r34*dv4
          t4 = eig1*r41*dv1 + eig2*r42*dv2 + eig3*r43*dv3 + eig4*r44*dv4
c
c Modify to calculate .5(fl +fr) from nodes
c instead of extrapolated ones
c
c         pL    = qnode(1,node1) 
c         uL    = qnode(2,node1) 
c         vL    = qnode(3,node1) 
c         wL    = qnode(4,node1) 
c         ubarL = xnorm*uL + ynorm*vL + znorm*wL
c
          fluxp1 = rlen*beta*ubarL
          fluxp2 = rlen*(uL*ubarL + xnorm*pL)
          fluxp3 = rlen*(vL*ubarL + ynorm*pL)
          fluxp4 = rlen*(wL*ubarL + znorm*pL)
c
c Now the right side
c
c         pR    = qnode(1,node2) 
c         uR    = qnode(2,node2) 
c         vR    = qnode(3,node2) 
c         wR    = qnode(4,node2) 
c         ubarR = xnorm*uR + ynorm*vR + znorm*wR
c
          fluxm1 = rlen*beta*ubarR
          fluxm2 = rlen*(uR*ubarR + xnorm*pR)
          fluxm3 = rlen*(vR*ubarR + ynorm*pR)
          fluxm4 = rlen*(wR*ubarR + znorm*pR)
c
          res1 = 0.5d0*(fluxp1 + fluxm1 - rlen*t1)
          res2 = 0.5d0*(fluxp2 + fluxm2 - rlen*t2)
          res3 = 0.5d0*(fluxp3 + fluxm3 - rlen*t3)
          res4 = 0.5d0*(fluxp4 + fluxm4 - rlen*t4)
c
c         call PLogFlops(318)
	  flops = flops + 318

c
          if (node1 .le. nnodes) then
           res(1,node1) = res(1,node1) + res1
           res(2,node1) = res(2,node1) + res2
           res(3,node1) = res(3,node1) + res3
           res(4,node1) = res(4,node1) + res4
c          call PLogFlops(4)
           flops = flops + 4
          endif
c
          if (node2 .le. nnodes) then
           res(1,node2) = res(1,node2) - res1
           res(2,node2) = res(2,node2) - res2
           res(3,node2) = res(3,node2) - res3
           res(4,node2) = res(4,node2) - res4
c          call PLogFlops(4)
           flops = flops + 4
          endif
        endif
c
      enddo
!$omp enddo
!$omp critical (residual_update)
#if defined(_OPENMP)
       do n = 1,nnodes
          resvec(1,n)=resvec(1,n)+res(1,n)
          resvec(2,n)=resvec(2,n)+res(2,n)
          resvec(3,n)=resvec(3,n)+res(3,n)
          resvec(4,n)=resvec(4,n)+res(4,n)
       enddo
       flops = flops + 4*nnodes
c      print *, 'For thread ', omp_get_thread_num(),', n is',n
#endif
!$omp end critical (residual_update)
!$omp end parallel  
#undef res(a,b)
#define res(a,b) resvec(a,b)
  
c     c68 = 6./8.
c     c18 = 1./8.
      c68 = 0.75d0
      c18 = 0.125d0
c
c Close contour over the boundaries 
c First do inviscid faces
c
      do 2020 n = 1, nnfacet
               node1 = isnode(f2ntn(n,1))
               node2 = isnode(f2ntn(n,2))
               node3 = isnode(f2ntn(n,3)) 

               x1 = x(node1)
               y1 = y(node1)
               z1 = z(node1)
               p1 = qnode(1,node1)

               x2 = x(node2)
               y2 = y(node2)
               z2 = z(node2)
               p2 = qnode(1,node2)

               x3 = x(node3)
               y3 = y(node3)
               z3 = z(node3)
               p3 = qnode(1,node3)
               
               ax = x2 - x1
               ay = y2 - y1
               az = z2 - z1

               bx = x3 - x1
               by = y3 - y1
               bz = z3 - z1
c
c Normal points away from grid interior.
c Magnitude is 1/3 area of surface triangle.
c
               xnorm =-0.5d0*(ay*bz - az*by)/3.d0
               ynorm = 0.5d0*(ax*bz - az*bx)/3.d0
               znorm =-0.5d0*(ax*by - ay*bx)/3.d0

               pa = c68*p1 + c18*(p2 + p3)
               pb = c68*p2 + c18*(p3 + p1)
               pc = c68*p3 + c18*(p1 + p2)
c
c              call PLogFlops(35)
               flops = flops + 35
               if (node1 .le. nnodes) then
                res(2,node1) = res(2,node1) + xnorm*pa
                res(3,node1) = res(3,node1) + ynorm*pa
                res(4,node1) = res(4,node1) + znorm*pa
c               call PLogFlops(6)
	        flops = flops + 6
               endif

               if (node2 .le. nnodes) then
                res(2,node2) = res(2,node2) + xnorm*pb
                res(3,node2) = res(3,node2) + ynorm*pb
                res(4,node2) = res(4,node2) + znorm*pb
c               call PLogFlops(6)
	        flops = flops + 6
               endif

               if (node3 .le. nnodes) then
                res(2,node3) = res(2,node3) + xnorm*pc
                res(3,node3) = res(3,node3) + ynorm*pc
                res(4,node3) = res(4,node3) + znorm*pc
c               call PLogFlops(6)
	        flops = flops + 6
               endif

 2020 continue
c
c Now viscous faces
c
      do 3020 n = 1, nvfacet
               node1 = ivnode(f2ntv(n,1))
               node2 = ivnode(f2ntv(n,2))
               node3 = ivnode(f2ntv(n,3)) 

               x1 = x(node1)
               y1 = y(node1)
               z1 = z(node1)
               p1 = qnode(1,node1)

               x2 = x(node2)
               y2 = y(node2)
               z2 = z(node2)
               p2 = qnode(1,node2)

               x3 = x(node3)
               y3 = y(node3)
               z3 = z(node3)
               p3 = qnode(1,node3)
               
               ax = x2 - x1
               ay = y2 - y1
               az = z2 - z1

               bx = x3 - x1
               by = y3 - y1
               bz = z3 - z1
c
c norm point away from grid interior.
c norm magnitude is 1/3 area of surface triangle.
c
               xnorm =-0.5d0*(ay*bz - az*by)/3.d0
               ynorm = 0.5d0*(ax*bz - az*bx)/3.d0
               znorm =-0.5d0*(ax*by - ay*bx)/3.d0

               pa = c68*p1 + c18*(p2 + p3)
               pb = c68*p2 + c18*(p3 + p1)
               pc = c68*p3 + c18*(p1 + p2)
c
c              call PLogFlops(35)
	       flops = flops + 35
c
               if (node1 .le. nnodes) then
                res(2,node1) = res(2,node1) + xnorm*pa
                res(3,node1) = res(3,node1) + ynorm*pa
                res(4,node1) = res(4,node1) + znorm*pa
c               call PLogFlops(6)
                flops = flops + 6

               endif

               if (node2 .le. nnodes) then
                res(2,node2) = res(2,node2) + xnorm*pb
                res(3,node2) = res(3,node2) + ynorm*pb
                res(4,node2) = res(4,node2) + znorm*pb
c               call PLogFlops(6)
                flops = flops + 6
               endif

               if (node3 .le. nnodes) then
                res(2,node3) = res(2,node3) + xnorm*pc
                res(3,node3) = res(3,node3) + ynorm*pc
                res(4,node3) = res(4,node3) + znorm*pc
c               call PLogFlops(6)
                flops = flops + 6
               endif

 3020 continue
c
c The next section of code is for when you dont care about
c preserving linear data on boundary. Also, doing this 
c matches the left hand side when not doing Newton-Krylov
c Usually just go around unless you are just experimenting
c
      goto 1025
c
c Loop over the boundaries
c First do inviscid nodes
c
c     do 1020 i = 1,nsnode
c       inode = isnode(i)
c
c       xnorm = sxn(i)
c       ynorm = syn(i)
c       znorm = szn(i)
c
c       p = qnode(1,inode)
c
c       if (inode .le. nnodes) then
c        res(2,inode) = res(2,inode) + xnorm*p
c        res(3,inode) = res(3,inode) + ynorm*p
c        res(4,inode) = res(4,inode) + znorm*p
c       endif
c
c1020 continue
c
c Now viscous nodes
c
c     do 1030 i = 1,nvnode
c       inode = ivnode(i)
c
c       xnorm = vxn(i)
c       ynorm = vyn(i)
c       znorm = vzn(i)
c
c       p = qnode(1,inode)
c
c       if (inode .le. nnodes) then
c        res(2,inode) = res(2,inode) + xnorm*p
c        res(3,inode) = res(3,inode) + ynorm*p
c        res(4,inode) = res(4,inode) + znorm*p
c       endif
c
c1030 continue
c
 1025 continue
c
c Now do far-field
c
CDIR$ IVDEP
       do 1040 i = 1,nfnode
         inode   = ifnode(i)
c
c Get normal and "other" 2 vectors. Remember that fxn,fyn and fzn 
c has the magnitude of the face contained in it.
c
         xnorm   = fxn(i)
         ynorm   = fyn(i)
         znorm   = fzn(i)
         area    = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
         xnorm   = xnorm/area
         ynorm   = ynorm/area
         znorm   = znorm/area
c
c Now lets get our other 2 vectors
c For first vector, use {1,0,0} and subtract off the component
c in the direction of the face normal. If the inner product of
c {1,0,0} is close to unity, use {0,1,0}
c
         dot = xnorm
         if(abs(dot).lt.0.95d0)then
          X1 = 1.d0 - dot*xnorm
          Y1 =    - dot*ynorm
          Z1 =    - dot*znorm
         else
          dot = ynorm
          X1 =    - dot*xnorm
          Y1 = 1.d0 - dot*ynorm
          Z1 =    - dot*znorm
         end if
c
c Normalize the first vector (V1)
c
         size = sqrt(X1*X1 + Y1*Y1 + Z1*Z1)
         X1 = X1/size
         Y1 = Y1/size
         Z1 = Z1/size
c
c Take cross-product of normal with V1 to get V2
c
         X2 = ynorm*Z1 - znorm*Y1
         Y2 = znorm*X1 - xnorm*Z1
         Z2 = xnorm*Y1 - ynorm*X1

c
c Calculate elements of T and T(inverse) evaluated at freestream
c
         ubar0 = xnorm*u0 + ynorm*v0 + znorm*w0
         c20   = ubar0*ubar0 + beta
         c0    = sqrt(c20)
         phi1  = xnorm*beta + u0*ubar0
         phi2  = ynorm*beta + v0*ubar0
         phi3  = znorm*beta + w0*ubar0
         phi4  = Y2*phi3 - Z2*phi2
         phi5  = Z2*phi1 - X2*phi3
         phi6  = X2*phi2 - Y2*phi1
         phi7  = Z1*phi2 - Y1*phi3
         phi8  = X1*phi3 - Z1*phi1
         phi9  = Y1*phi1 - X1*phi2

         t11 = 0.0d0
         t21 = X1
         t31 = Y1
         t41 = Z1

         t12 = 0.0d0
         t22 = X2
         t32 = Y2
         t42 = Z2

         t13 =  c0*beta
         t23 = xnorm*beta + u0*(ubar0 + c0)
         t33 = ynorm*beta + v0*(ubar0 + c0)
         t43 = znorm*beta + w0*(ubar0 + c0)

         t14 = -c0*beta
         t24 = xnorm*beta + u0*(ubar0 - c0)
         t34 = ynorm*beta + v0*(ubar0 - c0)
         t44 = znorm*beta + w0*(ubar0 - c0)

         ti11 = -(u0*phi4 + v0*phi5 + w0*phi6)/beta
         ti21 = -(u0*phi7 + v0*phi8 + w0*phi9)/beta
         ti31 =  .5d0*(c0 - ubar0)/beta
         ti41 = -.5d0*(c0 + ubar0)/beta
c
c Now, get the variables on the "inside"
c
         pi      = qnode(1,inode)
         ui      = qnode(2,inode)
         vi      = qnode(3,inode)
         wi      = qnode(4,inode)
         unorm   = xnorm*ui + ynorm*vi + znorm*wi
c
c If ubar is negative, take the reference condition from outside
c
c
         if(unorm.gt.0.0d0)then
          pr = pi
          ur = ui
          vr = vi
          wr = wi
         else
          pr = p0
          ur = u0
          vr = v0
          wr = w0
         end if
c
c Set rhs
c
         rhs1 = (ti11*pr + phi4*ur + phi5*vr + phi6*wr)/c20
         rhs2 = (ti21*pr + phi7*ur + phi8*vr + phi9*wr)/c20
         rhs3 = .5d0*(2.d0*ti31*pi + xnorm*ui + ynorm*vi + znorm*wi)/c20
         rhs4 = .5d0*(2.d0*ti41*p0 + xnorm*u0 + ynorm*v0 + znorm*w0)/c20
c
c Now do matrix multiplication to get values on boundary
c
         pb =                       t13*rhs3 + t14*rhs4
         ub = t21*rhs1 + t22*rhs2 + t23*rhs3 + t24*rhs4
         vb = t31*rhs1 + t32*rhs2 + t33*rhs3 + t34*rhs4
         wb = t41*rhs1 + t42*rhs2 + t43*rhs3 + t44*rhs4

         ubar = xnorm*ub + ynorm*vb + znorm*wb
c        call PLogFlops(180)
         flops = flops + 180
c
         if (inode .le. nnodes) then
          res(1,inode) = res(1,inode)+area*beta*ubar
          res(2,inode) = res(2,inode)+area*(ub*ubar + xnorm*pb)
          res(3,inode) = res(3,inode)+area*(vb*ubar + ynorm*pb)
          res(4,inode) = res(4,inode)+area*(wb*ubar + znorm*pb)
c         call PLogFlops(18)
          flops = flops + 18
         endif
c
 1040  continue
       call PLogFlops(flops)
c
c End of subroutine FLUX
c
      return
      end

c---------------------------------------------------------------
c The following subroutines are from node3t.f in the original
c code - D. K. Kaushik (1/17/97)
c---------------------------------------------------------------
c
c=============================== SUMGS ===============================72
c 
c Gets the weights for calculating gradients using least squares
c
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "SUMGS"
      subroutine SUMGS(nnodes,nedge,evec,xyz,
     1                 rxy,
     2                 irank,nvertices)

      implicit none      
#include "include/finclude/petscdef.h"
      integer nnodes,nedge,irank,nvertices
#if defined(INTERLACING)
      Scalar xyz(3,nvertices)
      Scalar rxy(7,nnodes)
      integer evec(2,nedge)
#define x(i) xyz(1,i)
#define y(i) xyz(2,i)
#define z(i) xyz(3,i)
#define r11(i) rxy(1,i)
#define r12(i) rxy(2,i)
#define r13(i) rxy(3,i)
#define r22(i) rxy(4,i)
#define r23(i) rxy(5,i)
#define r33(i) rxy(6,i)
#define r44(i) rxy(7,i)
#define eptr(j,i) evec(i,j)
#else
      Scalar xyz(nvertices,3)
      Scalar rxy(nnodes,7)
      integer evec(nedge,2)
#define x(i) xyz(i,1)
#define y(i) xyz(i,2)
#define z(i) xyz(i,3)
#define r11(i) rxy(i,1)
#define r12(i) rxy(i,2)
#define r13(i) rxy(i,3)
#define r22(i) rxy(i,4)
#define r23(i) rxy(i,5)
#define r33(i) rxy(i,6)
#define r44(i) rxy(i,7)
#define eptr(i,j) evec(i,j)
#endif
      integer i,n,node1,node2,ierr
      Scalar x1,y1,z1,x2,y2,z2,dx,dy,dz,dist,
     &       weight,w2,w11,w22,w33,
     &       r12r11,r13r11,r23r22,rmess

c
c Initialize all the rij to 0.0 
c
c     do 1000 i = 1,nnodes
c        r11(i) = 0.0
c        r12(i) = 0.0
c        r13(i) = 0.0
c        r22(i) = 0.0
c        r23(i) = 0.0
c        r33(i) = 0.0
c        r44(i) = 0.0
c1000 continue
      call PetscMemzero(rxy,8*7*nnodes,ierr)
      
c
c Now loop over the edges and accumulate the r's 
c
      do 1020 n = 1, nedge
c
          node1 = eptr(n,1)
          node2 = eptr(n,2)
c
          x1 = x(node1)
          y1 = y(node1)
          z1 = z(node1)
          x2 = x(node2)
          y2 = y(node2)
          z2 = z(node2)
c
          dx = x2 - x1
          dy = y2 - y1
          dz = z2 - z1
          dist = sqrt(dx*dx + dy*dy + dz*dz)
c         weight = 1.0d0/dist
          weight = 1.0d0
          w2 = weight*weight
c
          if (node1 .le. nnodes) then
           r11(node1) = r11(node1) + (x2 - x1)*(x2 - x1)*w2
           r12(node1) = r12(node1) + (x2 - x1)*(y2 - y1)*w2
           r13(node1) = r13(node1) + (x2 - x1)*(z2 - z1)*w2
          endif
c
          if (node2 .le. nnodes) then
           r11(node2) = r11(node2) + (x1 - x2)*(x1 - x2)*w2
           r12(node2) = r12(node2) + (x1 - x2)*(y1 - y2)*w2
           r13(node2) = r13(node2) + (x1 - x2)*(z1 - z2)*w2
          endif
 1020 continue
c /*
c Now calculate ||x|| = r11 by taking the square root
c Also divide r12 and r13 by ||x||
c */
      do 1030 i = 1,nnodes
        r11(i) = sqrt(r11(i))
        r12(i) = r12(i)/r11(i)
        r13(i) = r13(i)/r11(i)
 1030 continue
c /*
c Now calculate r22 and r23
c */
      do 1050 n = 1, nedge
c
          node1 = eptr(n,1)
          node2 = eptr(n,2)
c
          x1 = x(node1)
          y1 = y(node1)
          z1 = z(node1)
          x2 = x(node2)
          y2 = y(node2)
          z2 = z(node2)
c
          dx = x2 - x1
          dy = y2 - y1
          dz = z2 - z1
          dist = sqrt(dx*dx + dy*dy + dz*dz)
c         weight = 1.0/dist
          weight = 1.0d0
          dx = weight*dx
          dy = weight*dy
          dz = weight*dz
          w2 = weight*weight
c
          if (node1 .le. nnodes) then
           r22(node1) = r22(node1) +
     1                  (dy-dx*r12(node1)/r11(node1))**2
           r23(node1) = r23(node1) + dz*
     1                  (dy-dx*r12(node1)/r11(node1))
          endif
c
          if (node2 .le. nnodes) then
           r22(node2) = r22(node2) + 
     1                  (-dy + dx*r12(node2)/r11(node2))**2
           r23(node2) = r23(node2) - dz*
     1                  (-dy+dx*r12(node2)/r11(node2))
          endif
 1050 continue
c /*
c Now finish getting r22 and r23
c */
      do 1060 i = 1,nnodes
        r22(i) = sqrt(r22(i))
        r23(i) = r23(i)/r22(i)
 1060 continue
c /*
c Now all we have to do is get r33
c */
      do 1080 n = 1, nedge
c
          node1 = eptr(n,1)
          node2 = eptr(n,2)
c
          x1 = x(node1)
          y1 = y(node1)
          z1 = z(node1)
          x2 = x(node2)
          y2 = y(node2)
          z2 = z(node2)
c
          dx = x2 - x1
          dy = y2 - y1
          dz = z2 - z1
          dist = sqrt(dx*dx + dy*dy + dz*dz)
c         weight = 1.0/dist
          weight = 1.0d0
          dx = weight*dx
          dy = weight*dy
          dz = weight*dz
          w2 = weight*weight
c
          if (node1 .le. nnodes) then
           r33(node1) = r33(node1) + 
     1                  (dz-dx*r13(node1)/r11(node1)-
     2                  r23(node1)/r22(node1)*
     3                  (dy - dx*r12(node1)/
     4                   r11(node1)))**2
          endif
c
          if (node2 .le. nnodes) then
           r33(node2) = r33(node2) + 
     1                  (-dz+dx*r13(node2)/r11(node2)-
     2                  r23(node2)/r22(node2)*
     3                  (-dy + dx*r12(node2)/
     4                  r11(node2)))**2
          endif
c
 1080 continue
c /*
c Now just get the magnitude of r33
c */
      do 1090 i = 1,nnodes
        r33(i) = sqrt(r33(i))
 1090 continue
c
c Added by Dinesh Kaushik (6/27/97)
c /*The following addition changes the meaning of r11 .. r33. r44
c is the new parameter introduced by me. The new definitions
c are taken from LSTGS (where these parameters 
c are used finally).
c r11->w11
c r22->w22
c r33->w33
c r12->r12r11
c r13->r13r11
c r23->r23r22
c r44->rmess */

      do i = 1,nnodes
           w11 = 1.d0/(r11(i)*r11(i))
           w22 = 1.d0/(r22(i)*r22(i))
           w33 = 1.d0/(r33(i)*r33(i))
           r12r11 = r12(i)/r11(i)
           r13r11 = r13(i)/r11(i)
           r23r22 = r23(i)/r22(i)
           rmess=(r12(i)*r23(i)-
     1            r13(i)*r22(i))/
     2           (r11(i)*r22(i)*
     3            r33(i)*r33(i))

           r11(i) = w11
           r22(i) = w22
           r33(i) = w33
           r12(i) = r12r11
           r13(i) = r13r11
           r23(i) = r23r22
           r44(i) = rmess
      enddo
c Finished with SUMGS
c
      return
      end


c================================= LSTGS =============================72
c
c Calculates the Gradients at the nodes using weighted least squares
c This subroutine solves using Gram-Schmidt
c
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "LSTGS"
      subroutine LSTGS(nnodes,nedge,evec,
     1                 qvec,grad,xyz,
     2                 rxy,irank,nvertices)
c
      implicit none      
#include "include/finclude/petscdef.h"
      integer nnodes,nedge,irank,nvertices
c
#if defined(INTERLACING)
      Scalar qvec(4,nvertices)
      Scalar xyz(3,nvertices)
      Scalar rxy(7,nnodes)
      Scalar grad(3,4,nnodes)
      integer evec(2,nedge)
#define qnode(i,j) qvec(i,j)
#define gradx(i,j) grad(1,i,j)
#define grady(i,j) grad(2,i,j)
#define gradz(i,j) grad(3,i,j)
#define x(i) xyz(1,i)
#define y(i) xyz(2,i)
#define z(i) xyz(3,i)
#define r11(i) rxy(1,i)
#define r12(i) rxy(2,i)
#define r13(i) rxy(3,i)
#define r22(i) rxy(4,i)
#define r23(i) rxy(5,i)
#define r33(i) rxy(6,i)
#define r44(i) rxy(7,i)
#define eptr(j,i) evec(i,j)
#else
      Scalar qvec(nvertices,4)
      Scalar xyz(nvertices,3)
      Scalar rxy(nnodes,7)
      Scalar grad(nnodes,4,3)
      integer evec(nedge,2)
#define qnode(i,j) qvec(j,i)
#define gradx(i,j) grad(j,i,1)
#define grady(i,j) grad(j,i,2)
#define gradz(i,j) grad(j,i,3)
#define x(i) xyz(i,1)
#define y(i) xyz(i,2)
#define z(i) xyz(i,3)
#define r11(i) rxy(i,1)
#define r12(i) rxy(i,2)
#define r13(i) rxy(i,3)
#define r22(i) rxy(i,4)
#define r23(i) rxy(i,5)
#define r33(i) rxy(i,6)
#define r44(i) rxy(i,7)
#define eptr(i,j) evec(i,j)
#endif
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      Scalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,
     &        nstage,ncyct,iramp,nitfo
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo
      integer n,node1,node2,ierr
      integer flops
      Scalar  dx1,dy1,dz1,dx2,dy2,dz2,
     &        dq1,dq2,dq3,dq4,
     &        weight,w11,r12r11,r13r11,w22,r23r22,w33,rmess,
     &        coef1,coef2,termx,termy,termz
c     logging variables
c     integer flag
c     integer grad_event,node1_event,node2_event
c     character * 16 grad_label, node1_label, node2_label
c     data flag/-1/,grad_label/'GRAD            '/
c     data node1_label/'NODE1           '/
c     data node2_label/'NODE2           '/
c     save grad_event, grad_label, flag
c     save node1_event,node2_event,node1_label, node2_label

c     if (flag .eq. -1) then
c        call PLogEventRegister(grad_event,grad_label,'blue:',ierr)
c        call PLogEventRegister(node1_event,node1_label,'red:',ierr)
c        call PLogEventRegister(node2_event,node2_label,'green:',ierr)
c        flag = 1
c     endif
c     call PLogEventBegin(grad_event,0,0,0,0,ierr)

      flops = 0
c For checking out the code input a linear distribution
c
c     write(6,700)nnodes,nedge
c 700 format(1h ,'nnodes=',i5,' nedge=',i5)
c     do 1001 i = 1,nnodes
c     write(6,800)i,x(i),y(i),z(i)
c 800 format(1h ,'i x y z=',i5,3(f10.5,1x))
c       qnode(1,i) = 1.0*x(i) +  2.0*y(i) + 3.0*z(i)
c       qnode(2,i) = 3.0*x(i) +  4.0*y(i) + 6.0*z(i)
c       qnode(3,i) = 5.0*x(i) +  6.0*y(i) + 9.0*z(i)
c       qnode(4,i) = 7.0*x(i) +  8.0*y(i) + 12.0*z(i)
c       qnode(i,5) = 9.0*x(i) + 10.0*y(i) + 15.0*z(i)
c1001 continue
c
c Zero out the gradients
c
c     do 1000 i = 1,nnodes
c
c       gradx(1,i) = 0.0
c       grady(1,i) = 0.0
c       gradz(1,i) = 0.0
c
c       gradx(2,i) = 0.0
c       grady(2,i) = 0.0
c       gradz(2,i) = 0.0
c
c       gradx(3,i) = 0.0
c       grady(3,i) = 0.0
c       gradz(3,i) = 0.0
c
c       gradx(4,i) = 0.0
c       grady(4,i) = 0.0
c       gradz(4,i) = 0.0
c
c1000  continue
      call PetscMemzero(grad,12*8*nnodes,ierr)
c
c If second order, loop over all the faces accumulate sums
c
c     nitfo = 0
c     if(ntt.gt.nitfo.or.ivisc.gt.0)then
c     if (1 .gt. 0) then
       do 1020 n = 1, nedge
         node1 = eptr(n,1)
         node2 = eptr(n,2)
c        if ((node1 .le. nnodes).or.(node2 .le. nnodes)) then
           dx1 = x(node2) - x(node1)
           dy1 = y(node2) - y(node1)
           dz1 = z(node2) - z(node1)
c
c          dist = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
c          weight = 1.0/dist
           weight = 1.0d0
c          w2 = weight*weight
c
           dx1 = weight*dx1
           dy1 = weight*dy1
           dz1 = weight*dz1

           flops = flops + 6
c
c         call PLogEventBegin(node1_event,0,0,0,0,ierr)
          if (node1 .le. nnodes) then
           dq1 = weight*(qnode(1,node2) - qnode(1,node1))
           dq2 = weight*(qnode(2,node2) - qnode(2,node1))
           dq3 = weight*(qnode(3,node2) - qnode(3,node1))
           dq4 = weight*(qnode(4,node2) - qnode(4,node1))
c
c          w11 = 1./(r11(node1)*r11(node1))
c          w22 = 1./(r22(node1)*r22(node1))
c          w33 = 1./(r33(node1)*r33(node1))
c          r12r11 = r12(node1)/r11(node1)
c          r13r11 = r13(node1)/r11(node1)
c          r23r22 = r23(node1)/r22(node1)
c          rmess  = (r12(node1)*r23(node1) - r13(node1)*r22(node1))/
c    1              (r11(node1)*r22(node1)*r33(node1)*r33(node1))
c
           w11 = r11(node1)
           r12r11 = r12(node1)
           r13r11 = r13(node1)
           w22 = r22(node1)
           r23r22 = r23(node1)
           w33 = r33(node1)
           rmess  = r44(node1)
c
           coef1  = dy1 - dx1*r12r11
           coef2  = dz1 - dx1*r13r11 - r23r22*coef1
           termx = dx1*w11 - w22*r12r11*coef1 + rmess*coef2
           termy = w22*coef1 - r23r22*w33*coef2
           termz = w33*coef2
c
           gradx(1,node1) = gradx(1,node1) + termx*dq1
           grady(1,node1) = grady(1,node1) + termy*dq1
           gradz(1,node1) = gradz(1,node1) + termz*dq1
c
           gradx(2,node1) = gradx(2,node1) + termx*dq2
           grady(2,node1) = grady(2,node1) + termy*dq2
           gradz(2,node1) = gradz(2,node1) + termz*dq2
c
           gradx(3,node1) = gradx(3,node1) + termx*dq3
           grady(3,node1) = grady(3,node1) + termy*dq3
           gradz(3,node1) = gradz(3,node1) + termz*dq3
c
           gradx(4,node1) = gradx(4,node1) + termx*dq4
           grady(4,node1) = grady(4,node1) + termy*dq4
           gradz(4,node1) = gradz(4,node1) + termz*dq4
c
           flops = flops + 49
          endif
c         call PLogEventEnd(node1_event,0,0,0,0,ierr)
c
c Now do the other node
c
c         call PLogEventBegin(node2_event,0,0,0,0,ierr)
          if (node2 .le. nnodes) then
           dx2 = -dx1
           dy2 = -dy1
           dz2 = -dz1
c
           dq1 = weight*(qnode(1,node1) - qnode(1,node2))
           dq2 = weight*(qnode(2,node1) - qnode(2,node2))
           dq3 = weight*(qnode(3,node1) - qnode(3,node2))
           dq4 = weight*(qnode(4,node1) - qnode(4,node2))
c
c          w11 = 1./(r11(node2)*r11(node2))
c          w22 = 1./(r22(node2)*r22(node2))
c          w33 = 1./(r33(node2)*r33(node2))
c          r12r11 = r12(node2)/r11(node2)
c          r13r11 = r13(node2)/r11(node2)
c          r23r22 = r23(node2)/r22(node2)
c          rmess  = (r12(node2)*r23(node2) - r13(node2)*r22(node2))/
c    1              (r11(node2)*r22(node2)*r33(node2)*r33(node2))

           w11 = r11(node2)
           r12r11 = r12(node2)
           r13r11 = r13(node2)
           w22 = r22(node2)
           r23r22 = r23(node2)
           w33 = r33(node2)
           rmess  = r44(node2)
c
           coef1  = dy2 - dx2*r12r11
           coef2  = dz2 - dx2*r13r11 - r23r22*coef1
           termx = dx2*w11 - w22*r12r11*coef1 + rmess*coef2
           termy = w22*coef1 - r23r22*w33*coef2
           termz = w33*coef2
c
           gradx(1,node2) = gradx(1,node2) + termx*dq1
           grady(1,node2) = grady(1,node2) + termy*dq1
           gradz(1,node2) = gradz(1,node2) + termz*dq1
c
           gradx(2,node2) = gradx(2,node2) + termx*dq2
           grady(2,node2) = grady(2,node2) + termy*dq2
           gradz(2,node2) = gradz(2,node2) + termz*dq2
c
           gradx(3,node2) = gradx(3,node2) + termx*dq3
           grady(3,node2) = grady(3,node2) + termy*dq3
           gradz(3,node2) = gradz(3,node2) + termz*dq3
c
           gradx(4,node2) = gradx(4,node2) + termx*dq4
           grady(4,node2) = grady(4,node2) + termy*dq4
           gradz(4,node2) = gradz(4,node2) + termz*dq4
c
           flops = flops + 52
          endif
c         call PLogEventEnd(node2_event,0,0,0,0,ierr)
c       endif
c
 1020  continue
c     end if
      call PLogFlops(flops)
c     call PLogEventEnd(grad_event,0,0,0,0,ierr)
c
c End of LSTGS
c
      return
      end


c=================================== GETRES ==========================72
c
c Calculates the residual 
c Last Modified - D. K. Kaushik 1/23/97
c I have eliminated the input variables which were not needed - 
c dq, A, B, iupdate
c
c=====================================================================72
#undef __FUNC__
#define __FUNC__ "GETRES"
      subroutine GETRES(nnodes,ncell,nedge,nsface,nvface,nfface,nbface,
     1                  nsnode,nvnode,nfnode,isface,ivface,ifface,
     2                  ileast,isnode,ivnode,ifnode,
     &                  nnfacet,f2ntn,nnbound,
     &                  nvfacet,f2ntv,nvbound,
     &                  nffacet,f2ntf,nfbound,
     &                  evec,
     3                  sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,
     4                  xyzn,qvec,cdt,xyz,area,
     5                  grad,
     5                  resvec,resd,
     6                  turbre,slen,c2n,c2e,
     7                  us,vs,as,phi,
     &                  amut,ires,
     &                  irank, nvertices)
c
      implicit none      
#include "include/finclude/petscdef.h"
c
      integer nnodes, ncell, nedge,
     &        nbface,ileast,ires,
     &        nsface, nvface, nfface,
     &        nsnode, nvnode, nfnode,
     &        nnfacet,nvfacet,nffacet,
     &        nnbound,nvbound,nfbound,
     &        irank,nvertices
      integer evec(nedge,2)
      integer isface(1),ivface(1),ifface(1)
      integer isnode(1),ivnode(1),ifnode(1)
      integer c2n(ncell,4),c2e(ncell,6)
      integer f2ntn(nnfacet,4)
      integer f2ntv(nvfacet,4)
      integer f2ntf(nffacet,4)
c
      Scalar us(nbface,3,4),vs(nbface,3,4),as(nbface,3,4)
      Scalar sxn(1),syn(1),szn(1)
      Scalar vxn(1),vyn(1),vzn(1)
      Scalar fxn(1),fyn(1),fzn(1)
      Scalar xyz(nvertices,3),area(nvertices)
      Scalar xyzn(nedge,4)
      Scalar turbre(1),slen(1)
      Scalar qvec(nvertices,4),resd(4,nnodes)
      Scalar phi(nvertices,4)
      Scalar cdt(nvertices)
      Scalar grad(3,4,nvertices)
c     real dq(nnodes,4)
c     real r11(nvertices),r12(nvertices),r13(nvertices)
c     real r22(nvertices),r23(nvertices),r33(nvertices)
      Scalar amut(nnodes)
c
      integer i
      Scalar  title(20),beta,alpha,Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      Scalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,
     &        nstage,ncyct,iramp,nitfo
      Scalar  gtol
      integer icycle,nsrch,ilu0,ifcn
      Scalar  rms(1001),clw(1001),cdw(1001),
     &        cmw(1001),xres(1001)
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo
      common/gmcom/gtol,icycle,nsrch,ilu0,ifcn
      common/history/rms,clw,cdw,cmw,xres
C     logging variables
c     integer flux_event, delta2_event, flag
c     character * 16 flux_label, delta2_label
c     data flag/-1/,flux_label/'FLUX            '/
c     data delta2_label/'DELTA2          '/
c     save flux_event,delta2_event,flag, flux_label,delta2_label
#if defined(INTERLACING)
       Scalar resvec(4,nnodes)
#define res(i,j) resvec(i,j)
#else
       Scalar resvec(nnodes,4)
#define res(i,j) resvec(j,i)
#endif
c

c     if (flag .eq. -1) then
c        call PLogEventRegister(delta2_event,delta2_label,'red:',ierr)
c        call PLogEventRegister(flux_event,flux_label,'blue:',ierr)         
c        flag = 1
c     endif
c
c Calculate the time step
c
      if(ires.eq.1) goto 888
c     call PLogEventBegin(delta2_event,0,0,0,0,ierr)
      call DELTAT2(nnodes,nedge,qvec,cdt,
     &             xyz,area,xyzn,evec,
     &             sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,
     &             nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
     &             irank,nvertices)
c     call PLogEventEnd(delta2_event,0,0,0,0,ierr)
  888 continue
c /*'
c   Calculate the gradients 
c   ----Kyle seems to recommend only LSTGS for gradients,
c   so I have commented the GETGRAD call - DKK (1/17/97)
c
c     if (ileast.eq.0) then
c        call GETGRAD(nnodes,ncell,nedge,nsface,nvface,nfface,
c    &                isface,ivface,ifface,eptr,ncolor,ncount,
c    &                qnode,gradx,grady,x,y,
c    &                area,wx,wy,xn,yn,rl)
c
c     else if (ileast.eq.4) then
c     if (ileast.eq.4) then
c        call LSTGS(nnodes,nedge,eptr,
c    1              qnode,gradx,grady,gradz,xyz,
c    2              r11,r12,r13,r22,r23,r33,irank,nvertices)
c     end if
c
c zero out residuals (viscous residuals are zeroed in vfluxnew)
c '*/
      do 1002 i = 1,nnodes
          res(1,i)=0.0d0
          res(2,i)=0.0d0
          res(3,i)=0.0d0
          res(4,i)=0.0d0

          phi(i,1)=1.0d0
          phi(i,2)=1.0d0
          phi(i,3)=1.0d0
          phi(i,4)=1.0d0
 1002   continue
c
c /*'
c If not doing Newton-Krylov and iflim=1 call the Flux Limiter
c
c     if(iflim.eq.1.and.ifcn.ne.1)then
c      call TIMLIM(nnodes,nedge,qnode,res,dq,phi,ncolor,ncount,
c    &             gradx,grady,gradz,x,y,z,eptr)
c '*/
c If we used the limiter we need to zero out the residual again
c since we used it for scratch space
c
c     do 1003 i = 1,nnodes
c         res(1,i)=0.0
c         res(2,i)=0.0
c         res(3,i)=0.0
c         res(4,i)=0.0
c1003   continue
c
c     end if
c
c   Split the fluxes and perform the flux balance
c
c       call PLogEventBegin(flux_event,0,0,0,0,ierr)

         call FLUX(nnodes,ncell,nedge,
     &            nsface,nvface,nfface,isface,ivface,ifface,
     &            nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
     &            nnfacet,f2ntn,nnbound,
     &            nvfacet,f2ntv,nvbound,
     &            nffacet,f2ntf,nfbound,
     &            grad,evec,qvec,
     &            xyz,resvec,resd,xyzn,sxn,syn,szn,vxn,vyn,vzn,
     &            fxn,fyn,fzn,phi,irank,nvertices)
c        call PLogEventEnd(flux_event,0,0,0,0,ierr)
c /*'
c calculate viscous fluxes
c
c     if (ivisc.gt.0) then
c           call VISRHS (nnodes,ncell,nedge,
c           call EDGEVIS(nnodes,ncell,nedge,
c    &                  nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
c    &                  nsface,nvface,nfface,isface,ivface,ifface,
c    &                  nnfacet,f2ntn,nnbound,ncolorn,countn,
c    &                  nvfacet,f2ntv,nvbound,ncolorv,countv,
c    &                  nffacet,f2ntf,nfbound,ncolorf,countf,
c    &                  nccolor,nccount,
c    &                  eptr,c2n,c2e,
c    &                  sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,
c    &                  x,y,z,gradx,grady,gradz,
c    &                  qnode,amut,res,phi)
c     end if
c '*/
c End of subroutine GETRES
c
      return
      end

c---------------------------------------------------------------
c The following subroutine is from node4t.f in the original
c code - D. K. Kaushik (1/17/97)
c---------------------------------------------------------------
c
c=============================================================================
c
c  Opens files for I/O
c
c=============================================================================
#undef __FUNC__
#define __FUNC__ "OPENM"
      SUBROUTINE OPENM(irank)
c
c  TAPE7  -- input:  mach number, angle of attack etc.. 
c  TAPE9  -- input:  reads restart file
c  TAPE10 -- output: residual history
c  TAPE11 -- output: writes restart file
c  TAPE12 -- output: writes residual and lift for plotting
c  TAPE13 -- output: writes flowfield for contour plotting
c
      implicit none
      integer  irank
      OPEN(UNIT= 7,FILE='testgrid/ginput.faces',
     +form='formatted',STATUS='OLD')
 
c     OPEN(UNIT=9,FILE='framer.bin',
c    +form='unformatted',STATUS='old')
 
      if (irank .eq. 0) OPEN(UNIT= 10,FILE='frame.out3',
     +form='formatted',STATUS='unknown')
 
c     OPEN(UNIT= 11,FILE='frame.bin',
c    +form='unformatted',STATUS='unknown')
 
c     OPEN(UNIT= 12,FILE='frame.plt',
c    +form='formatted',STATUS='unknown')
 
c     OPEN(UNIT= 13,FILE='frame.tec',
c    +form='formatted',STATUS='unknown')
 
c     OPEN(UNIT= 14,FILE='frame.fast.g',
c    +form='unformatted',STATUS='unknown')
 
c     OPEN(UNIT= 15,FILE='frame.fast.q',
c    +form='unformatted',STATUS='unknown')

      return
      end
c
c
c===================================================================
c
c Get the IA, JA, and IAU arrays
c
c===================================================================
#undef __FUNC__
#define __FUNC__ "GETIA"
      subroutine GETIA(nnodes,nedge,evec,ia,ideg,irank)
      implicit none      
#include "include/finclude/petscdef.h"
      integer nnodes,nedge,irank
      integer ia(1),ideg(1)
#if defined(INTERLACING)
       integer evec(2,nedge)
#define eptr(j,i) evec(i,j)
#else
       integer evec(nedge,2)
#define eptr(i,j) evec(i,j)
#endif
       integer i,node1,node2
c
c First get the degree of each node using ideg as a dummy array
c
      do 1000 i = 1,nnodes
        ideg(i) = 0
 1000 continue
c
      do 1010 i = 1,nedge
        node1 = eptr(i,1)
        node2 = eptr(i,2)
        if (node1 .le. nnodes) ideg(node1) = ideg(node1) + 1
        if (node2 .le. nnodes) ideg(node2) = ideg(node2) + 1
 1010 continue
c
c Now we can fill the ia array fairly easily
c
      ia(1) = 1
      do 1020 i = 1,nnodes
        ia(i+1) = ia(i) + ideg(i) + 1
c       write(9,100)i,ideg(i)
c 100   format(1h ,'deg(',i6,')=',i6)
 1020 continue
c
      return
      end
c===================================================================
c
c Get the IA, JA, and IAU arrays
c
c===================================================================
#undef __FUNC__
#define __FUNC__ "GETJA"
      subroutine GETJA(nnodes,nedge,evec,ia,ja,iwork,irank)
      implicit none      
#include "include/finclude/petscdef.h"
      integer nnodes,nedge,irank
      integer ia(1),ja(1),iwork(1)
#if defined(INTERLACING)
       integer evec(2,nedge)
#define eptr(j,i) evec(i,j)
#else
       integer evec(nedge,2)
#define eptr(i,j) evec(i,j)
#endif
       integer i,index,node1,node2,index1,index2,
     &         istart,iend
c     open(unit=90,file='map.out',status='UNKNOWN')
c
c Now we need to get the JA array
c First fill the diagonal places
c
      do 1040 i = 1,nnodes
        index = ia(i)
        ja(index) = i
        iwork(i) = 1
 1040 continue
c
      do 1030 i = 1,nedge
        node1 = eptr(i,1)
        node2 = eptr(i,2)
c
        if (node1 .le. nnodes) then
          index1 = ia(node1) + iwork(node1) 
          iwork(node1) = iwork(node1) + 1
          ja(index1) = node2
        endif
        if (node2 .le. nnodes) then
          index2 = ia(node2) + iwork(node2)
          iwork(node2) = iwork(node2) + 1
          ja(index2) = node1
        endif
 1030 continue
c
c Now lets sort all our "bins" and get the correct one on the diagonal
c
      do 1050 i = 1,nnodes
        istart = ia(i)
        iend   = ia(i+1) - 1
c       write(9,200)i,istart,iend
c 200   format(1h ,'Sorting ',i6,' istart iend = ',i6,1x,i6)
        call SORTER(istart,iend,ja,i)
 1050 continue
c
c Now get the "fhelp" array which will assist in assembling
c the flux Jacobians into the correct location in the alu array
c
c     write(90,*) 'fhelp array'
c     do 1060 i = 1,nedge
c       node1 = eptr(i,1)
c       node2 = eptr(i,2)
c
c First take care of node1
c
c       idiag = iau(node1)
c
c If the offdiagonal term is ordered later in the ja array 
c
c       if(node2.gt.node1)then
c        jstart = idiag + 1
c        jend   = ia(node1+1) - 1
c       else
c        jstart = ia(node1)
c        jend   = idiag -1
c       end if
c
c        do 1070 j = jstart,jend
c          if(ja(j).eq.node2)fhelp(i,1) = j
c1070    continue
c
c
c Now take care of node2
c
c       idiag = iau(node2)
c
c If the offdiagonal term is ordered later in the ja array 
c
c       if(node1.gt.node2)then
c        jstart = idiag + 1
c        jend   = ia(node2+1) - 1
c       else
c        jstart = ia(node2)
c        jend   = idiag -1
c       end if
c
c       do 1080 j = jstart,jend
c         if(ja(j).eq.node1)fhelp(i,2) = j
c1080   continue
c       write(90,*) i,fhelp(i,1),fhelp(i,2)
c1060 continue
c      close(90)
c
      return
      end
c
c
c===================================================================
c
c Sort each of our bins
c
c===================================================================
#undef __FUNC__
#define __FUNC__ "SORTER"
      subroutine SORTER(istart,iend,ja,inode)
      implicit none      
#include "include/finclude/petscdef.h"
      integer istart,iend,inode
      integer ja(1)
c
      integer min,minsave,jsave,i,j
      do 1000 i = istart,iend
        min = ja(i)
        minsave = ja(i)
        jsave = i
        do 1010 j = i+1,iend
          if(ja(j).lt.min)then
            min = ja(j)
            jsave = j
          end if
 1010   continue
        ja(i) = min
        ja(jsave) = minsave
c       if(ja(i).eq.inode)iau(inode) = i
 1000 continue
c
      return
      end
c
c===================================================================
#undef __FUNC__
#define __FUNC__ "IREAD"
      subroutine IREAD(unit,n,iper,arr)
c===================================================================
      implicit none      
#include "include/finclude/petscdef.h"
      integer unit, n, iper
      integer arr(n,iper)
      integer i
      if (iper .eq. 1) then
         read(unit) (arr(i,1), i = 1, n)
      else if (iper .eq. 2) then
         read(unit) (arr(i,1), i = 1, n),(arr(i,2), i = 1, n)
      else if (iper .eq. 3) then
         read(unit) (arr(i,1), i = 1, n),(arr(i,2), i = 1, n),
     1              (arr(i,3), i = 1, n)
      else if (iper .eq. 4) then
         read(unit) (arr(i,1), i = 1, n),(arr(i,2), i = 1, n),
     1              (arr(i,3), i = 1, n), (arr(i,4), i = 1, n)
      endif
      return
      end
c
c===================================================================
#undef __FUNC__
#define __FUNC__ "RREAD"
      subroutine RREAD(unit,n,iper,arr)
c===================================================================
      implicit none      
#include "include/finclude/petscdef.h"
      integer unit, n, iper
      Scalar arr(n,iper)
      integer i
      if (iper .eq. 1) then
         read(unit) (arr(i,1), i = 1, n)
      else if (iper .eq. 2) then
         read(unit) (arr(i,1), i = 1, n),(arr(i,2), i = 1, n)
      else if (iper .eq. 3) then
         read(unit) (arr(i,1), i = 1, n),(arr(i,2), i = 1, n),
     1              (arr(i,3), i = 1, n)
      else if (iper .eq. 4) then
         read(unit) (arr(i,1), i = 1, n),(arr(i,2), i = 1, n),
     1              (arr(i,3), i = 1, n), (arr(i,4), i = 1, n)
      endif
      return
      end


