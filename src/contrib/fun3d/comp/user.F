!---------------------------------------------------------------
! The following subroutines are from node2t.f in the original
! code - D. K. Kaushik (1/17/97)
!---------------------------------------------------------------
!
!
! 2-D Navier Stokes on Unstructured Grids
!
!================================ SCLOCK ====================================
!
!  SCLOCK get the CPU time
!
!============================================================================
      subroutine SCLOCK(time)

!      time = second(time)
      time = 1.0

      return
      end


!============================== FORLINK ==============================72
!
!  FORLINK establishes links between FORTRAN common blocks and C
!
!=====================================================================72
      subroutine FORLINK()

      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,         &
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,          &
     &             nstage,ncyct,iramp,nitfo,ncyc
      common/gmcom/gtol,icycle,nsrch,ilu0,ifcn
      common/refgeom/sref,cref,bref,xmc,ymc,zmc

      call CLINK(title,cfl1,gtol,refgeom)
!
! End of subroutine FORLINK
!
      return
      end

!============================== Block_Initialization  ================72
!
! Initializes the common blocks members for turbulence model
!
!=====================================================================72
      block data Block_Initialization
      common/turb/vkar,cmu,ce1,ce2,aplus1,aplus2,turbinf
      common/spalrt/cb1,sig,cb2,cw1,cw2,cw3,cv1,ct1,ct2,ct3,ct4
      data vkar,cmu,ce1,ce2/0.41,0.09,1.2,2.0 /
      data aplus1,aplus2,turbinf/26.0,10.0,0.1 /
      data cb1,sig,cb2,cw2,cw3/0.1355,0.66667,0.622,0.3,2.0/
!/*
! Comment out old coefficients
!      data cv1,ct1,ct2,ct3,ct4/7.1,1.0,2.0,1.1,2.0/
!*/
       data cv1,ct1,ct2,ct3,ct4/7.1,1.0,2.0,1.2,0.5/

      end

!================================== INIT =============================72
!
! Initializes the flow field
!
!=====================================================================72
      subroutine INIT(nnodes, qvec, turbre, amut,nvnode, ivnode,irank)

      dimension turbre(1),amut(1)
      integer ivnode(1)

      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,         &
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/turb/vkar,cmu,ce1,ce2,aplus1,aplus2,turbinf
      common/spalrt/cb1,sig,cb2,cw1,cw2,cw3,cv1,ct1,ct2,ct3,ct4
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,          &
     &             nstage,ncyct,iramp,nitfo,ncyc
!
!     data vkar,cmu,ce1,ce2/0.41,0.09,1.2,2.0 /
!     data aplus1,aplus2,turbinf/26.0,10.0,0.1 /
!     data cb1,sig,cb2,cw2,cw3/0.1355,0.66667,0.622,0.3,2.0/
c
! Comment out old coefficients
!      data cv1,ct1,ct2,ct3,ct4/7.1,1.0,2.0,1.1,2.0/
c
!      data cv1,ct1,ct2,ct3,ct4/7.1,1.0,2.0,1.2,0.5/
c
c
#if defined(INTERLACING)
       real qvec(5,nnodes)
#define qnode(i,j) qvec(i,j)
#else
       real qvec(nnodes,5)
#define qnode(i,j) qvec(j,i)
#endif
!
      cw1 = cb1/vkar/vkar + (1. + cb2)/sig
      if(ivisc.gt.4)turbinf = 1.341946
!      if(ivisc.gt.4)turbinf = 0.5
      if(ivisc.gt.4.and.itran.eq.1)turbinf = 0.1
!
! Note that for Spalarts model, I use turbinf as the freestream value of
! the dependent variable just as in the Baldwin-Barth model.
! The constant is set so that in the freestream, nu_t=0.009 (1.341946)
!
      res0 = 1.0
      resc = 1.0
c
      gamma = 1.4
      gm1   = gamma - 1.0
      gp1   = gamma + 1.0
      gm1g  = gm1/gamma
      gp1g  = gp1/gamma
      ggm1  = gamma*gm1
 
      pi = 4.*atan(1.)
      conv = 180./pi
      rho0  = 1.0
      c0    = 1.0
      p0    = rho0*c0*c0 / gamma
!
!
!  The following u0,v0,w0 are Kyle's original (no yaw) components
!
!     u0 = xmach * cos( alpha/conv )
!     v0 = xmach * sin( alpha/conv )
!     w0 = 0.0
!
!  The following u0,v0,w0 are CFL3D's yawed components, based
!  on info from Bob Biedron.  CFL3D has z "up" and y "out the
!  span".
!
!     u0 = xmach*cos(alpha)*cos(beta)
!     w0 = xmach*sin(alpha)*cos(beta)
!     v0 = -xmach*sin(beta)
!
!  However, in FUN3D, we have y "up" and z "out the span".  So
!  we think that the components should be like:
!
!   u0(FUN3D) =   u0(CFL3D)
!   v0(FUN3D) =   w0(CFL3D)
!   w0(FUN3D) = - v0(CFL3D)
!
!     u0 = xmach * cos(alpha/conv)*cos(yaw/conv)
!     v0 = xmach * sin(alpha/conv)*cos(yaw/conv)
!     w0 = xmach * sin(yaw/conv)
!
!  But NOW (11/4/96), Kyle wants to follow CFL3D's grid-axis
!  convention, so let's set it up that way.
!
#if defined(CFL3D_AXIS)    
      u0 =   xmach * cos(alpha/conv) * cos(yaw/conv)
      v0 = - xmach * sin(yaw/conv)
      w0 =   xmach * sin(alpha/conv) * cos(yaw/conv)
#else
      u0 = xmach * cos( alpha/conv )
      v0 = xmach * sin( alpha/conv )
      w0 = 0.0
#endif
!
      ei0   = p0/(( gamma - 1.0 )*rho0)
      et0   = rho0*( ei0 + 0.5*(u0*u0 + v0*v0 + w0*w0))
      h0    = (et0 + p0)/rho0
!      pt0   = p0*(1.0 + 0.5*gm1*xmach*xmach)**3.5
      pt0 = 1.0
      if (irank .eq. 0) write(10,500)rho0,c0,p0,u0,v0,
     1                               w0,ei0,et0,h0,pt0
  500 format(1h ,'Initial values: rho,c,p,u,v,w,ei,e,h,pt ',11f10.5) 
      
      do n = 1,nnodes
         qnode(1,n) = rho0
         qnode(2,n) = rho0*u0
         qnode(3,n) = rho0*v0
         qnode(4,n) = rho0*w0
         qnode(5,n) = et0
      enddo
!     This if statement has been introduced to save memory for
!     inviscid and laminar cases -- D. K. Kaushik (12/20/97)
      if (ivisc.ge.3) then
       do n = 1,nnodes
         turbre(n)  = 0.0
         amut(n)    = 0.0
       enddo
      endif
!
! If viscous, zero out the velocity on the surface
! and set the energy so it reflects the correct wall temp
!
      Prandtl = 0.72
      Twall = 1.0
      Twall = 1.0 + .5*sqrt(Prandtl)*gm1*xmach*xmach
!     print *, "Just Before Viscous"
!
      do 9010 i = 1,nvnode
!
! Compute the velocity normal to the surface
!
        k       = ivnode(i)
!
! Set the velocity to zero energy according to specified wall temperature
!
        qnode(2,k) = 0.0
        qnode(3,k) = 0.0
        qnode(4,k) = 0.0
        qnode(5,k) = qnode(1,k)*Twall/ggm1
!
 9010 continue
!
! If turbulent, initialize turbre
!
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
            rmu = 1.0
            chi = turbre(n)/rmu
            fv1 = chi**3/(chi**3 + cv1**3)
            amut(n)   = fv1*turbinf
 1020    continue
      end if
!     print *, "I am out of INIT"
 
      return
  110 format(1h ,'Parameters for turbulence model')
  120 format(1h ,'k=',f10.5,' cmu=',f10.5,' ce1=',f10.5,'ce2=',f10.5)
  130 format(1h ,'aplus1',f10.5,' aplus2=',f10.5,' turbinf=',f10.5)
  140 format(1h ,'k=',f10.5,' cb1=',f10.5,' sig=',f10.5,'cb2=',f10.5)
  150 format(1h ,'cw1=',f10.5,' cw2=',f10.5,' cw3=',f10.5,' cv1=',f10.5)
  160 format(1h ,'ct1=',f10.5,' ct2=',f10.5,' ct3=',f10.5,' ct4=',f10.5)
!
! End of subroutine INIT
!
      
      end


c================================ READR1 ====================================
c
!  Reads input parameters
c
c============================================================================
      subroutine READR1(ileast, irank)

      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo,ncyc
      common/refgeom/sref,cref,bref,xmc,ymc,zmc

      read(7,10)(title(i),i=1,20)
      if (irank .eq. 0) write(10,11)(title(i),i=1,20)

      read(7,10)

      read(7,24)mseq,ihane,ivisc,ileast,iflim,jupdate
      if (irank .eq. 0) write(10,25)mseq,ihane,ivisc
      if (irank .eq. 0) write(10,28)ileast,iflim,jupdate

      read(7,10)

      read(7,12)xmach,alpha,yaw,Re
      if (irank .eq. 0) write(10,13)xmach,alpha,yaw,Re

      read(7,10)

      read(7,14)sref,cref,bref,xmc,ymc,zmc
      if (irank .eq. 0) write(10,15)sref,cref,bref,xmc,ymc,zmc

      read(7,10)

      read(7,26)cfl2,dt,irest,itran,nbtran
      if (irank .eq. 0) write(10,27)cfl2,dt,irest

      if (irank .eq. 0) then
       if(ivisc.eq.5.or.ivisc.eq.6) write(10,123)itran,nbtran
      endif

   10 format(20a4)
   11 format(1h ,20a4)
   12 format(3f10.5,e14.7)
   13 format(1h ,'Xmach = ',f10.5,' Alpha = ',f10.5,' Yaw = ',f10.5,
     &       ' Re = ',e14.7)
   14 format(6f10.5)
   15 format(1h ,'sref = ',f10.5,' cref = ',f10.5,' bref = ',f10.5,
     &' xmc = ',f10.5,' ymc = ',f10.5,' zmc = ',f10.5)
   24 format(i10,i10,i10,i10,i10,i10)
   25 format(1h ,'mseq = ',i3,' ihane = ',i3,' ivisc=',i3)
   26 format(2f10.5,3i10)
   27 format(1h ,' cfl2= ',e14.7,' dt= ',f10.5,'irest= ',i5)
   28 format(1h ,'ileast= ',i5,' iflim= ',i5,' jupdate= ',i5)
  123 format(1h ,'itran = ',i5,' nbtran= ',i5)

      return
      end


c================================ RDGPAR =============================72
c
! Reads grid parameters
! I am using this subroutine in modified form - DKK (1/8/97)
c
c=====================================================================72
      subroutine RDGPAR(nnodes,ncell,nedge,
     &                  ncolort,ncolore,
     &                  nnbound,nvbound,nfbound,
     &                  nnfacet,nvfacet,nffacet,
     &                  nsnode,nvnode,nfnode,ntte,
     &                  nsface,nvface,nfface,
     &                  irank)

      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb

      character*40 filename

      integer unit
! 
!  DEBUG: to check things out
!  
      nsface = 0
      nvface = 0
      nfface = 0
c
!  end DEBUG: to check things out
c

      unit =  20

!     read(7,*) filename
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
! Read grid 
c
c=====================================================================72
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

      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb

      integer unit

      integer evec(nedge,2)

      real sxn(nsnode),syn(nsnode),szn(nsnode)
      real vxn(nvnode),vyn(nvnode),vzn(nvnode)
      real fxn(nfnode),fyn(nfnode),fzn(nfnode)
      real vol(nnodes),slen(nnodes)
      real x(nnodes),y(nnodes),z(nnodes)
      real xn(nedge),yn(nedge),zn(nedge),ra(nedge)
c
!  Garbage variables
c
      integer c2n(ncell,4), c2e(ncell,6)
      integer nntet(nnbound), nnpts(nnbound)
      integer nvtet(nvbound), nvpts(nvbound)
      integer nftet(nfbound), nfpts(nfbound)
      integer f2ntn(nnfacet,4), f2ntv(nvfacet,4), f2ntf(nffacet,4)
      integer isnode(nsnode),ivnode(nvnode),ifnode(nfnode)
      integer idum, jdum

      unit = 20
c
!  Read Cell to Node indices
c
      if (ivisc.ne.0) then
         read (unit) (c2n(j,1),j=1,ncell),(c2n(j,2),j=1,ncell),
     &               (c2n(j,3),j=1,ncell),(c2n(j,4),j=1,ncell)
      else
         read (unit) (idum,j=1,ncell),(idum,j=1,ncell),
     &               (idum,j=1,ncell),(idum,j=1,ncell)
      end if

c
!  Read Cell to Edge indices
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
!  Read coordinates of grid points
c
      read (unit) (x(i),i=1,nnodes)
      read (unit) (y(i),i=1,nnodes)
      read (unit) (z(i),i=1,nnodes)
c
!  Read median dual volume surrounding each node
c
      read (unit) (vol(i),i=1,nnodes)
c
!  Read node indices for each edge
c
      read (unit) (evec(k,1),k=1,nedge),(evec(k,2),k=1,nedge)
c
!  Read in unit normals of dual mesh face (from node 1 to node 2)
!  and the area of the dual mesh face 
c
      read (unit) (xn(k),k=1,nedge),(yn(k),k=1,nedge),
     &            (zn(k),k=1,nedge),(ra(k),k=1,nedge)
c
!  Read Cell colors
c
      read (unit) (idum,i=1,nccolor)
c
!  Read edge colors
c
      read (unit) (idum,i=1,ncolor)
c
!  Read solid boundary face colors
c
      do i=1, nnbound
         read (unit) idum,(jdum,j=1,idum)
      end do
c
!  Read viscous boundary face colors
c
      do i=1, nvbound
         read (unit) idum,(jdum,j=1,idum)
      end do
c
!  Read boundary face colors
c
      do i=1, nfbound
         read (unit) idum,(jdum,j=1,idum)
      end do
c
!  Read inviscid surface info
c
      read (unit) (nntet(i),i=1,nnbound)
      read (unit) (nnpts(i),i=1,nnbound)
c
!  Read viscous surface info
c
      read (unit) (nvtet(i),i=1,nvbound)
      read (unit) (nvpts(i),i=1,nvbound)
c
!  Read far field info
c
      read (unit) (nftet(i),i=1,nfbound)
      read (unit) (nfpts(i),i=1,nfbound)
c
!  Read list of faces on inviscid boundaries
c
      read (unit) (f2ntn(i,1),i=1,nnfacet),(f2ntn(i,2),i=1,nnfacet),
     &            (f2ntn(i,3),i=1,nnfacet),(f2ntn(i,4),i=1,nnfacet)
c
!  Read list of faces on viscous boundaries
c
      read (unit) (f2ntv(i,1),i=1,nvfacet),(f2ntv(i,2),i=1,nvfacet),
     &            (f2ntv(i,3),i=1,nvfacet),(f2ntv(i,4),i=1,nvfacet)
c
!  Read list of faces on far field boundaries
c
      read (unit) (f2ntf(i,1),i=1,nffacet),(f2ntf(i,2),i=1,nffacet),
     &            (f2ntf(i,3),i=1,nffacet),(f2ntf(i,4),i=1,nffacet)
c
!  Read list of info associated with nodes on each inviscid boundary
c
      read (unit) (isnode(i),i=1,nsnode)
      read (unit) (sxn(i),i=1,nsnode),(syn(i),i=1,nsnode),
     &            (szn(i),i=1,nsnode)
c
!  Read list of info associated with nodes on each viscous boundary
c
      read (unit) (ivnode(i),i=1,nvnode)
      read (unit) (vxn(i),i=1,nvnode),(vyn(i),i=1,nvnode),
     &            (vzn(i),i=1,nvnode)
c
!  Read list of info associated with nodes on each far field boundary
c
      read (unit) (ifnode(i),i=1,nfnode)
      read (unit) (fxn(i),i=1,nfnode),(fyn(i),i=1,nfnode),
     &            (fzn(i),i=1,nfnode)
c
! Read in the distance function
c
      if (ivisc.gt.3) then
         read(unit)(slen(i),i=1,nnodes)
      end if
 
      close (unit)
c
!  End of subroutine README
c
      return
      end


!================================ RREST ==============================72
!
! Reads input data for restarts 
!
!=====================================================================72
      subroutine RREST(nnodes, qvec, turbre, amut)

      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/history/rms(3000),clw(3000),cdw(3000),cmxw(3000),
     2               cmyw(3000),cmzw(3000),cxw(3000),cyw(3000),
     3               czw(3000),xres(3000)
      common/turb/vkar,cmu,ce1,ce2,aplus1,aplus2,turbinf
      common/spalrt/cb1,sig,cb2,cw1,cw2,cw3,cv1,ct1,ct2,ct3,ct4

      dimension qvec(5,nnodes), turbre(nnodes), amut(nnodes)

      read (9) qvec

      if (ivisc.ge.1) then
         read(9)turbre
         read(9)amut
      end if

      close(9)

      if (irest.eq.2) then
         do 1000 i = 1,nnodes
            turbre(i) = turbinf
 1000    continue
      end if

      return
c
! End of subroutine RREST
c
      end


c================================ WREST ==============================72
c
! Writes output flowfield for restarts 
c
c=====================================================================72
      subroutine WREST(nnodes, qvec, turbre, amut)

      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/history/rms(3000),clw(3000),cdw(3000),cmxw(3000),
     2               cmyw(3000),cmzw(3000),cxw(3000),cyw(3000),
     3               czw(3000),xres(3000)

      dimension qvec(5,nnodes), turbre(nnodes), amut(nnodes)

      write (11) qvec

      if (ivisc.ge.1) then
         write (11) turbre
         write (11) amut
      end if

      close(11)

      return
c
! End of subroutine WREST
c
      end


c================================ PLLAN ====================================
c
!  Writes output for plotting 
c
c============================================================================
      subroutine PLLAN(nnodes,irank)

      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/history/rms(3000),clw(3000),cdw(3000),cmxw(3000),
     2               cmyw(3000),cmzw(3000),cxw(3000),cyw(3000),
     3               czw(3000),xres(3000)
!
! Write out the residual history
!
!     tpnpi = (xres(ntt) - xres(1))/float(ntt*nnodes)
      tpnpi = xres(ntt)/float(ntt*nnodes)
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
!        xres(n) = xres(n)/1.0e6
         write(12,100)n,rms(n),clw(n),cdw(n),cmzw(n),xres(n)
 1200 continue
  100 format(1h ,i5,1x,e14.7,2x,e14.7,2x,e14.7,2x,e14.7,2x,e14.7)
  300 format(1h ,'#time/node/iteration:',e14.7)
  301 format(1h ,'   v1::Iteration')
  302 format(1h ,'   v2::Log(R)')
  303 format(1h ,'   v3::c_l')
  304 format(1h ,'   v4::c_d')
  305 format(1h ,'   v5::c_m')
  306 format(1h ,'   v6::CPU time')
!
! End of subroutine PLLAN
!
      return
      end


!================================ TECFLO =============================72
!
!  Writes a formatted file that contains an input file for TECPLOT
!
!=====================================================================72
      subroutine TECFLO(nnodes,
     &                  nnbound,
     3                  nnfacet,
     &                  nsnode,
     4                  x,y,z,qvec,
     5                  nnpts,nntet,
     6                  f2ntn,isnode,
     7                  istep,irank,iopen,iclose,ibnd)
!
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0

      integer nntet(1),nnpts(1)
      integer f2ntn(nnfacet,4)
      integer isnode(1)
      integer i,j,n,i1,i2,i3,ic,isp,ist
      integer istep, iopen, iclose, ibnd
 
      real x(1),y(1),z(1)
 
      character c4*4,title*20
      character aaa*10
      save aaa
#if defined(INTERLACING)
       real qvec(5,nsnode)
#define qnode(i,j) qvec(i,j)
#else
       real qvec(nsnode,5)
#define qnode(i,j) qvec(j,i)
#endif
      if (iopen .eq. 1) then
       write(aaa,'(4hflow,i4.4)') istep
       open(unit=99,file=aaa,status='unknown')
!
!   + write tecplot header
!
!     write(99,'(a)') 'TITLE= "',title,'"'
!
c================================== INIT =============================72
       write(99,'(a)') 'VARIABLES="X     ","Y     ","Z     ","RHO   ",', $  
     $      '"U     ","V     ","W     ","P/Pinf","S     ","Mach  "'  
!
      endif
!
      call ETOP(nsnode,qvec)

      pinf = p0
!
!   + do a zone-title, so we can keep track of things
!   + start with solid-wall boundary surfaces
!  
      isp   = 1
      ist   = 1
      do 10 i=1,nnbound
!
!        write(c4,"(i4)") i
!       if (i .ge.    0 .and. i .le.    9) ic = 4
!        if (i .ge.   10 .and. i .le.   99) ic = 3
!        if (i .ge.  100 .and. i .le.  999) ic = 2
!        if (i .ge. 1000 .and. i .le. 9999) ic = 1
!
        if (ibnd .eq. 1) then
          write(c4,'(3hnn.,i1.1)') i
          write(99,1000) c4,nnpts(i),nntet(i)
        else if (ibnd .eq. 2) then
          write(c4,'(3hnv.,i1.1)') i
          write(99,1000) c4,nnpts(i),nntet(i)
        else if (ibnd .eq. 3) then
          write(c4,'(3hff.,i1.1)') i
          write(99,1000) c4,nnpts(i),nntet(i)
	endif
!
        write(99,1010) (x(j),j=isp,isp+nnpts(i)-1)
        write(99,1010) (y(j),j=isp,isp+nnpts(i)-1)
        write(99,1010) (z(j),j=isp,isp+nnpts(i)-1)
        write(99,1010) (qnode(1,j),j=isp,isp+nnpts(i)-1)
        write(99,1010) (qnode(2,j),j=isp,isp+nnpts(i)-1)
        write(99,1010) (qnode(3,j),j=isp,isp+nnpts(i)-1)
        write(99,1010) (qnode(4,j),j=isp,isp+nnpts(i)-1)           
        write(99,1010) (qnode(5,j)/pinf,j=isp,isp+nnpts(i)-1)
        write(99,1010) ((qnode(5,j)/pinf)/
     1                  qnode(1,j)**gamma - 1.0,   
     2                  j=isp,isp+nnpts(i)-1)
        write(99,1010) (sqrt((qnode(2,j)**2 + 
     1                  qnode(3,j)**2 + 
     2                  qnode(4,j)**2)/
     3                  (gamma*qnode(5,j)/
     4                  qnode(1,j))),
     5                  j=isp,isp+nnpts(i)-1)
!
        do 30 j=ist,ist+nntet(i)-1
!
          i1 = f2ntn(j,1) - isp + 1
          i2 = f2ntn(j,2) - isp + 1
          i3 = f2ntn(j,3) - isp + 1
!
          write(99,1020) i1,i2,i3,i3
!
   30   continue
!
        isp = isp + nnpts(i) 
        ist = ist + nntet(i)
!
   10 continue
      call PTOE(nsnode,qvec)
!
!     End of subroutine tecflo
!
 1000 format('ZONE T="',a,'", I=',i6,', J=',i6,', F=FEBLOCK')
 1010 format(1P10E13.5)
 1020 format(4I10)
      if (iclose .eq. 1) then
        print *, 'Field values written to the file ',aaa
        close(99)
!1030   format('Field values written to the file ',a)
      endif
      return
      end


!=============================== FASFLO ==============================72
!
! Write a FAST binary file
!
!=====================================================================72
      subroutine FASFLO(nnodes,nsnode,nnfacet,isnode,f2ntn,x,y,z,qvec)

      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb

      integer isnode(1),f2ntn(nnfacet,1)

      real qvec(5,nnodes)
      real x(1),y(1),z(1)

      nfaces=nnfacet
      time=1.0

      write (14) nsnode,nfaces,0
      write (14) (x(isnode(i)),i=1,nsnode),
     &           (y(isnode(i)),i=1,nsnode),
     &           (z(isnode(i)),i=1,nsnode),
     &           ((f2ntn(i,j),j=1,3),i=1,nnfacet),
     &           (1,i=1,nnfacet)

      write (15) nsnode,1,1
!     write (15) beta,alpha,Re,time
      write (15) (1.0,i=1,nsnode),
     &           ((qvec(nx,isnode(i)),nx=2,4),i=1,nsnode),
     &            (qvec(isnode(i),1),i=1,nsnode)

!
! End of subroutine FASFLO
!
      return
      end

!================================ ETOH ===============================72
!
!  Converts total energy to total enthalpy
!  Also converts rhou and rhov to u and v
!
!=====================================================================72
      subroutine ETOH(nvertices, qvec)
#include "finclude/petscsys.h"

      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      integer ierr
      PetscLogDouble flops
#if defined(INTERLACING)
      real qvec(5,nvertices) 
#define qnode(i,j) qvec(i,j)
#else
      real qvec(nvertices,5) 
#define qnode(i,j) qvec(j,i)
#endif
      flops = 0.0
      do 1000 i = 1,nvertices
         rho = qnode(1,i)
         u   = qnode(2,i)/rho
         v   = qnode(3,i)/rho
         w   = qnode(4,i)/rho
         q2  = u*u + v*v + w*w
         E   = qnode(5,i)
         p   = gm1*(E - 0.5*rho*q2)
 
         qnode(2,i) = u
         qnode(3,i) = v
         qnode(4,i) = w
         qnode(5,i) = (E + p)/rho        
 1000 continue
      flops = flops + nvertices*14
      call PetscLogFlops(flops,ierr)
c
! End of subroutine ETOH
c
      return
      end


!================================ HTOE ===============================72
!
!  Converts total enthalpy to total energy
!  Also converts u and v back to rhou and rhov
!
!=====================================================================72
      subroutine HTOE(nvertices, qvec)

      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      integer ierr
      PetscLogDouble flops
#if defined(INTERLACING)
      real qvec(5,nvertices) 
#define qnode(i,j) qvec(i,j)
#else
      real qvec(nvertices,5) 
#define qnode(i,j) qvec(j,i)
#endif

      flops = 0.0
      do 1000 i = 1,nvertices
         rho = qnode(1,i)
         u   = qnode(2,i)
         v   = qnode(3,i)
         w   = qnode(4,i)
         q2  = u*u + v*v + w*w
         H   = qnode(5,i)
 
         qnode(2,i) = rho*u
         qnode(3,i) = rho*v
         qnode(4,i) = rho*w
         qnode(5,i) = rho*(H + 0.5*gm1*q2)/gamma
 1000 continue
      flops = flops + nvertices*13.0
      call PetscLogFlops(flops,ierr)
c
! End of subroutine HTOE
c
      return
      end


c================================ ETOP ===============================72
c
!  Converts total energy to pressure
!  Also converts rhou and rhov to u and v
c
c=====================================================================72
      subroutine ETOP(nvertices, qvec)
      integer ierr
      PetscLogDouble flops
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
#if defined(INTERLACING)
      real qvec(5,nvertices) 
#define qnode(i,j) qvec(i,j)
#else
      real qvec(nvertices,5) 
#define qnode(i,j) qvec(j,i)
#endif

      flops = 0.0
      do 1000 i = 1,nvertices
         rho = qnode(1,i)
         u   = qnode(2,i)/rho
         v   = qnode(3,i)/rho
         w   = qnode(4,i)/rho
         q2  = u*u + v*v + w*w
         E   = qnode(5,i)
         p   = gm1*(E - 0.5*rho*q2)
 
         qnode(2,i) = u
         qnode(3,i) = v
         qnode(4,i) = w
         qnode(5,i) = p
 1000 continue
      flops = flops + nvertices*12.0
      call PetscLogFlops(flops,ierr)
c
! End of subroutine ETOP
c
      return
      end


c================================ PTOE ===============================72
c
!  Converts pressure to total energy
!  Also converts u and v back to rhou and rhov
c
c=====================================================================72
      subroutine PTOE(nvertices, qvec)
      integer ierr
      PetscLogDouble flops
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1

#if defined(INTERLACING)
      real qvec(5,nvertices) 
#define qnode(i,j) qvec(i,j)
#else
      real qvec(nvertices,5) 
#define qnode(i,j) qvec(j,i)
#endif
      flops = 0.0
      do 1000 i = 1,nvertices
         rho = qnode(1,i)
         u   = qnode(2,i)
         v   = qnode(3,i)
         w   = qnode(4,i)
         q2  = u*u + v*v + w*w
         P   = qnode(5,i)
 
         qnode(2,i) = rho*u
         qnode(3,i) = rho*v
         qnode(4,i) = rho*w
         qnode(5,i) = P/gm1 + 0.5*rho*q2
 1000 continue
      flops = flops + nvertices*12.0
      call PetscLogFlops(flops,ierr)
c
! End of subroutine PTOE
c
      return
      end

c================================ L2NORM =============================72
c
!  calculates the L2 norm of the residual
!  I have eliminated dq - D. K. Kaushik (1/23/97)
c=====================================================================72
      subroutine L2NORM(res,nnodes,nnode_glo,x,y,z,vol,irank)
#include "finclude/petscsys.h"
 
      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/history/rms(3000),clw(3000),cdw(3000),cmxw(3000),
     2               cmyw(3000),cmzw(3000),cxw(3000),cyw(3000),
     3               czw(3000),xres(3000)
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo,ncyc

      dimension res(4,nnodes),x(1),y(1),z(1)
!     dimension dq(nnodes,4),vol(1)
      dimension vol(1)

      integer irank, icount
      real sum, sumglo

c
! Set the flag to monitor which type of residual you want
! iflag=0 "normal" residual
! iflag=1 residual/volume
! iflag=2 dq
c
      iflag = 0
 
      t1 = float(nnode_glo)
      sum = 0.0
      rmax = 0.0
c
! iflag=0
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
! iflag=1
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
! iflag=2
c
!     if (iflag.eq.2)then
!        do 1030 i = 1,nnodes
!           sum = sum + dq(i,1)*dq(i,1)
!           test = abs(dq(i,1))
!           if (test.ge.rmax)then
!              rmax = test
!              xloc = x(i)
c              yloc = y(i)
!              zloc = z(i)
!           end if
c1030    continue
!     end if
 
!     rms(ntt) = sqrt(sum/t1)
      icount = 1
      sumglo=0.0
!     write(*,*) 'I am in L2NORM just before Reduce op'
!     write(*,*) 'Local res. norm on processor ',irank,' is ',sqrt(sum)
      call MPI_ALLREDUCE(sum,sumglo,icount,
     >                   MPIU_SCALAR, MPI_SUM,
     >                   MPI_COMM_WORLD,ierr)
!     write(*,*) 'In L2NORM - residual norm is', sqrt(sumglo)
      rms(ntt) = sqrt(sumglo/t1)
c
! If we are ramping the cfl with SER save res0 and resc
c
      if (ntt.eq.1) res0 = rms(ntt)
      resc = rms(ntt)
      compare = res0*cfl1/cfl2
      if (resc.lt.compare) resc = compare
      ratio = res0/resc

!     write(6,100)ntt,rms(ntt),rmax,xloc,yloc,zloc,ratio
      if (irank .eq. 0) write(10,100)ntt,rms(ntt),rmax,
     >                 xloc,yloc,zloc,ratio

  100 format(1h ,'Iteration',i6,' rms rmax xloc yloc zloc rat = ',
     &       6(1e14.7,1x))
  202 format(1h ,2x,e14.7,2x,e14.7,2x,i6)
c
! End of subroutine L2NORM
c
      return
      end


!================================ FORCE  ============================72
!
!  Provided by Bob Biedron, inserted by Eric on 10/31/96
!  (Note:  "history" common block has to be modified everywhere
!   to use this routine!)
!
!  NOTE:  This currently ONLY DOES PRESSURE FORCES correctly.
!         There's no guarantee on the viscous force stuff.
!         (We think it's right though)
!
!   sign conventions for forces and moments:
!     cmx....moment about x-axis through the moment center (xmc,ymc,zmc)
!            positive value for ccw moment when viewed from +x axis
!     cmy....moment about y-axis through the moment center (xmc,ymc,zmc)
!            positive value for ccw moment when viewed from +y axis
!     cmz....moment about z-axis through the moment center (xmc,ymc,zmc)
!            positive value for ccw moment when viewed from +z axis
!     cx.....x-component of force
!            positive for force in +x direction
!     cy.....y-component of force
!            positive for force in +y direction
!     cz.....z-component of force
!            positive for force in +z direction
!
!  Modified - D. K. Kaushik (1/3/98)
!  Orginal name of this subroutine was FORCE2.
!  Added new parameters - clift, cdrag, cmom, irank, nvertices
!  
!=====================================================================72
      subroutine FORCE(nnodes,nedge,
     &                 isnode,ivnode,
     &                 nnfacet,f2ntn,nnbound,
     &                 nvfacet,f2ntv,nvbound,
     &                 evec,qvec,
     &                 x,y,z,
     &                 nvnode,c2n,ncell,
     &                 amut,
     &                 sface_bit,vface_bit,
     &                 clift, cdrag, cmom,irank,nvertices)
#include "finclude/petscsys.h"

      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/history/rms(3000),clw(3000),cdw(3000),cmxw(3000),
     2               cmyw(3000),cmzw(3000),cxw(3000),cyw(3000),
     3               czw(3000),xres(3000)
      common/refgeom/sref,cref,bref,xmc,ymc,zmc
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo,ncyc
      integer isnode(1),ivnode(1)
      integer f2ntn(nnfacet,4)
      integer f2ntv(nvfacet,4)
      integer c2n(ncell,4)
      integer sface_bit(nnfacet), vface_bit(nvfacet)
      integer icount,ierr

      real x(1),y(1),z(1)
      real amut(1)
      real cl_loc,cd_loc,cx_loc,cy_loc,cz_loc,cmx_loc,cmy_loc,cmz_loc
      real cl_glo,cd_glo,cx_glo,cy_glo,cz_glo,cmx_glo,cmy_glo,cmz_glo
      real coef_loc(8), coef_glo(8)
      PetscSizeT sizeofscalar
c 
#if defined(INTERLACING)
      real qvec(5,nvertices)
      integer evec(2,nedge)
#define qnode(i,j) qvec(i,j)
#define eptr(j,i) evec(i,j)
#else
      real qvec(nvertices,5)
      integer evec(nedge,2)
#define qnode(i,j) qvec(j,i)
#define eptr(i,j) evec(i,j)
#endif
c
      pi = 4.*atan(1.)
      conv = 180./pi
      csa=cos(alpha/conv)
      sna=sin(alpha/conv)
      csy=cos(yaw/conv)
      sny=sin(yaw/conv)
c
c  initialize forces to zero  
c
      cl_loc = 0.0
      cd_loc = 0.0
      cx_loc = 0.0
      cy_loc = 0.0
      cz_loc = 0.0
      cmx_loc = 0.0
      cmy_loc = 0.0
      cmz_loc = 0.0
      call PetscDataTypeGetSize(PETSC_SCALAR,sizeofscalar,ierr)
      call PetscMemzero(coef_glo,sizeofscalar*8,ierr)
 
c
c     if(ntt.eq.ncyc) then
c       if(nnbound.gt.0) then
c         write(16,*) 'Inviscid bodies:'
c       endif
c     endif
c
      do n = 1, nnfacet
        if (sface_bit(n) .eq. 1) then
               node1 = isnode(f2ntn(n,1))
               node2 = isnode(f2ntn(n,2))
               node3 = isnode(f2ntn(n,3))

               x1    = x(node1)
               y1    = y(node1)
               z1    = z(node1)

               x2    = x(node2)
               y2    = y(node2)
               z2    = z(node2)

               x3    = x(node3)
               y3    = y(node3)
               z3    = z(node3)
               
               ax = x2 - x1
               ay = y2 - y1
               az = z2 - z1

               bx = x3 - x1
               by = y3 - y1
               bz = z3 - z1
!
!  norm points outward, away from grid interior.
!  norm magnitude is area of surface triangle.
!
               xnorm =-0.5*(ay*bz - az*by)
               ynorm = 0.5*(ax*bz - az*bx)
               znorm =-0.5*(ax*by - ay*bx)
 
               rho1  = qnode(1,node1)
               u1    = qnode(2,node1)/rho1
               v1    = qnode(3,node1)/rho1
               w1    = qnode(4,node1)/rho1
               p1    = gm1*(qnode(5,node1) 
     &               - .5*rho1*(u1*u1 + v1*v1 + w1*w1))
               rho2  = qnode(1,node2)
               u2    = qnode(2,node2)/rho2
               v2    = qnode(3,node2)/rho2
               w2    = qnode(4,node2)/rho2
               p2    = gm1*(qnode(5,node2) 
     &               - .5*rho2*(u2*u2 + v2*v2 + w2*w2))
               rho3  = qnode(1,node3)
               u3    = qnode(2,node3)/rho3
               v3    = qnode(3,node3)/rho3
               w3    = qnode(4,node3)/rho3
               p3    = gm1*(qnode(5,node3) 
     &               - .5*rho3*(u3*u3 + v3*v3 + w3*w3))
 
               press = (p1 + p2 + p3)/3.0
               cp    = 2.*(press/p0-1.)/(gamma*xmach*xmach)        
c
               dcx = cp*xnorm
               dcy = cp*ynorm
               dcz = cp*znorm

               xmid = (x1 + x2 + x3)/3.0
               ymid = (y1 + y2 + y3)/3.0
               zmid = (z1 + z2 + z3)/3.0

c
c  In the following force and moment summations, the lines that
c  are commented out are the original FUN3D summations.  The new
c  ones are from CFL3D.
c
c              clw(ntt) = clw(ntt) - dcx*sna + dcy*csa
c              cdw(ntt) = cdw(ntt) + dcx*csa + dcy*sna
c
c              cmxw(ntt) = cmxw(ntt)
c    &                   + (ymid - yr)*dcz - (zmid - zr)*dcy
c              cmyw(ntt) = cmyw(ntt)
c    &                   - (xmid - xr)*dcz + (zmid - zr)*dcx
c              cmzw(ntt) = cmzw(ntt)
c    &                   + (xmid - xr)*dcy - (ymid - yr)*dcx
c
#if defined(CFL3D_AXIS)    
                cl_loc = cl_loc - dcx*sna     + dcz*csa
                cd_loc = cd_loc + dcx*csa*csy - dcy*sny
     &                          + dcz*sna*csy
c
                cx_loc = cx_loc + dcx
                cy_loc = cy_loc + dcy
                cz_loc = cz_loc + dcz
c
                cmx_loc = cmx_loc + dcz*(ymid-ymc)
     &                            - dcy*(zmid-zmc)
                cmy_loc = cmy_loc - dcz*(xmid-xmc)
     &                            + dcx*(zmid-zmc)
                cmz_loc = cmz_loc + dcy*(xmid-xmc)
     &                            - dcx*(ymid-ymc)
#else
                cl_loc = cl_loc - dcx*sna + dcy*csa
                cd_loc = cd_loc + dcx*csa + dcy*sna
c
                cx_loc = cx_loc + dcx
                cy_loc = cy_loc + dcy
                cz_loc = cz_loc + dcz
c
                cmx_loc = cmx_loc + dcz*(ymid-ymc)
     &                            - dcy*(zmid-zmc)
                cmy_loc = cmy_loc - dcz*(xmid-xmc)
     &                            + dcx*(zmid-zmc)
                cmz_loc = cmz_loc + dcy*(xmid-xmc)
     &                            - dcx*(ymid-ymc)
#endif
c
         endif

       enddo
c
c Viscous boundary
c
c     nstart=1
c
c     if(ntt.eq.ncyc) then
c       if(nvbound.gt.0) then
c         write(16,*) 'Viscous bodies:'
c       endif
c     endif
c
c     do 40 i=1,nvbound
c
c Initialize temporary sums for this body:
c Pressure components for both Cl and Cd
c Viscous components for both Cl and Cd
c
c       clp   = 0.
c       cdp   = 0.
        clv   = 0.
        cdv   = 0.
c
c  First compute the lift and drag due to viscous forces
c  for this boundary and add to the total lift and drag
c
c       call SKINFRIC(ivnode,f2ntv,c2n,nnodes,
c    &                x,y,z,qvec,amut,nvbound,nvnode,nvfacet,ncell,
c    &                clv,cdv,vface_bit,irank,nvertices)
c
c  Add it
c
        cl_loc = cl_loc + clv
        cd_loc = cd_loc + cdv
c
       do n = 1, nvfacet
         if (vface_bit(n) .eq. 1) then
               node1 = ivnode(f2ntv(n,1))
               node2 = ivnode(f2ntv(n,2))
               node3 = ivnode(f2ntv(n,3))

               x1    = x(node1)
               y1    = y(node1)
               z1    = z(node1)

               x2    = x(node2)
               y2    = y(node2)
               z2    = z(node2)

               x3   = x(node3)
               y3   = y(node3)
               z3   = z(node3)
               
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
               xnorm =-0.5*(ay*bz - az*by)
               ynorm = 0.5*(ax*bz - az*bx)
               znorm =-0.5*(ax*by - ay*bx)
 
               rho1  = qnode(1,node1)
               u1    = qnode(2,node1)/rho1
               v1    = qnode(3,node1)/rho1
               w1    = qnode(4,node1)/rho1
               p1    = gm1*(qnode(5,node1) 
     &               - .5*rho1*(u1*u1 + v1*v1 + w1*w1))
               rho2  = qnode(1,node2)
               u2    = qnode(2,node2)/rho2
               v2    = qnode(3,node2)/rho2
               w2    = qnode(4,node2)/rho2
               p2    = gm1*(qnode(5,node2) 
     &               - .5*rho2*(u2*u2 + v2*v2 + w2*w2))
               rho3  = qnode(1,node3)
               u3    = qnode(2,node3)/rho3
               v3    = qnode(3,node3)/rho3
               w3    = qnode(4,node3)/rho3
               p3    = gm1*(qnode(5,node3) 
     &               - .5*rho3*(u3*u3 + v3*v3 + w3*w3))
 
               press = (p1 + p2 + p3)/3.0
               cp    = 2.*(press/p0-1.)/(gamma*xmach*xmach)        
c
               dcx = cp*xnorm
               dcy = cp*ynorm
               dcz = cp*znorm
c
               xmid = (x1 + x2 + x3)/3.
               ymid = (y1 + y2 + y3)/3.
               zmid = (z1 + z2 + z3)/3.
c
c  In the following force and moment summations, the lines that
c  are commented out are the original FUN3D summations.  The new
c  ones are from CFL3D.
c
c               clw(ntt) = clw(ntt) - dcx*sna + dcy*csa
c               cdw(ntt) = cdw(ntt) + dcx*csa + dcy*sna
c
c               cmxw(ntt) = cmxw(ntt)
c    &                    + (ymid - yr)*dcz - (zmid - zr)*dcy
c               cmyw(ntt) = cmyw(ntt)
c    &                    - (xmid - xr)*dcz + (zmid - zr)*dcx
c               cmzw(ntt) = cmzw(ntt)
c    &                    + (xmid - xr)*dcy - (ymid - yr)*dcx
c 
#if defined(CFL3D_AXIS)    
                cl_loc = cl_loc - dcx*sna     + dcz*csa
                cd_loc = cd_loc + dcx*csa*csy - dcy*sny
     &                          + dcz*sna*csy
c
                cx_loc = cx_loc + dcx
                cy_loc = cy_loc + dcy
                cz_loc = cz_loc + dcz
c
                cmx_loc = cmx_loc + dcz*(ymid-ymc)
     &                            - dcy*(zmid-zmc)
                cmy_loc = cmy_loc - dcz*(xmid-xmc)
     &                            + dcx*(zmid-zmc)
                cmz_loc = cmz_loc + dcy*(xmid-xmc)
     &                            - dcx*(ymid-ymc)
#else
                cl_loc = cl_loc - dcx*sna + dcy*csa
                cd_loc = cd_loc + dcx*csa + dcy*sna
c
                cx_loc = cx_loc + dcx
                cy_loc = cy_loc + dcy
                cz_loc = cz_loc + dcz
c
                cmx_loc = cmx_loc + dcz*(ymid-ymc)
     &                            - dcy*(zmid-zmc)
                cmy_loc = cmy_loc - dcz*(xmid-xmc)
     &                            + dcx*(zmid-zmc)
                cmz_loc = cmz_loc + dcy*(xmid-xmc)
     &                            - dcx*(ymid-ymc)
#endif
c
         endif
 
       enddo
c
      icount = 8
      coef_loc(1) = cl_loc
      coef_loc(2) = cd_loc
      coef_loc(3) = cx_loc
      coef_loc(4) = cy_loc
      coef_loc(5) = cz_loc
      coef_loc(6) = cmx_loc
      coef_loc(7) = cmy_loc
      coef_loc(8) = cmz_loc
      call MPI_ALLREDUCE(coef_loc,coef_glo,icount,
     >                   MPIU_SCALAR, MPI_SUM,
     >                   MPI_COMM_WORLD,ierr)
!     call MPI_ALLREDUCE(cd_loc,cd_glo,icount,
!    >                   MPI_DOUBLE_PRECISION, MPI_SUM,
!    >                   MPI_COMM_WORLD,ierr)
!     call MPI_ALLREDUCE(cm_loc,cm_glo,icount,
!    >                   MPI_DOUBLE_PRECISION, MPI_SUM,
!    >                   MPI_COMM_WORLD,ierr)
!
!  Now finally let's non-dimensionalize our coefficients using
!  our reference geometry and CFL3D's conventions
!
       clw(ntt)  = coef_glo(1) / sref
       cdw(ntt)  = coef_glo(2) / sref
       cxw(ntt)  = coef_glo(3) / sref
       cyw(ntt)  = coef_glo(4) / sref
       czw(ntt)  = coef_glo(5) / sref
       cmxw(ntt) = coef_glo(6) / (sref*bref)
       cmyw(ntt) = coef_glo(7) / (sref*cref)
       cmzw(ntt) = coef_glo(8) / (sref*bref)
       clift = clw(ntt)
       cdrag = cdw(ntt)
#if defined(CFL3D_AXIS)    
       cmom  = cmzw(ntt)
#else
       cmom  = cmyw(ntt)
#endif
c
c Update the timing information - Added by D. K. Kaushik (1/17/97)
      xres(ntt) = tot
c End of subroutine FORCE
c
      return
      end
!
!================================ SKINFRIC ===========================72
!
!  This gets skin friction drag (and contribution to lift) for all 
!  viscous boundaries --- modified D. K. Kaushik (1/10/98)
!
!=====================================================================72
      subroutine SKINFRIC(ivnode,f2ntv,c2n,nnodes,
     &                    x,y,z,qvec,amut,nvbound,nvnode,nvfacet,ncell,
     &                    clv,cdv,vface_bit,irank,nvertices)
!
      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/refgeom/sref,cref,bref,xmc,ymc,zmc
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo,ncyc
!
      real clv, cdv, pi
      real x(1),y(1),z(1)
      real amut(1)
      real nx1,nx2,nx3,nx4
      real ny1,ny2,ny3,ny4
      real nz1,nz2,nz3,nz4
      real nxd, nyd, nzd
      real nxl, nyl, nzl
!
      integer ivnode(nvnode),f2ntv(nvfacet,4),vface_bit(1)
      integer c2n(ncell,4)
! 
#if defined(INTERLACING)
      real qvec(5,nvertices)
#define qnode(i,j) qvec(i,j)
#else
      real qvec(nvertices,5)
#define qnode(i,j) qvec(j,i)
#endif
!
!  Some constants
!
      c43 = 4./3.
      c23 = 2./3.
      xmr = xmach / Re
      pi = acos(-1.)
      conv = 180. / pi 
!
!  Initialize sums
!
      clv  = 0.0
      cdv  = 0.0
!
! Stuff for laminar viscosity
!
       Tref = 460.
       cstar = 198.6/Tref
!
      do n = 1, nvfacet
        if (vface_bit(n) .eq. 1) then
          node1 = ivnode(f2ntv(n,1))
          node2 = ivnode(f2ntv(n,2))
          node3 = ivnode(f2ntv(n,3))
!
          icell = f2ntv(n,4)
!
          node4 = c2n(icell,1) + c2n(icell,2) + c2n(icell,3)
     &          + c2n(icell,4) - node1 - node2 - node3
!
          x1 = x(node1)
          y1 = y(node1)
          z1 = z(node1)
!
          x2 = x(node2)
          y2 = y(node2)
          z2 = z(node2)
!
          x3 = x(node3)
          y3 = y(node3)
          z3 = z(node3)
!
          x4 = x(node4)
          y4 = y(node4)
          z4 = z(node4)
!
!   Lets get outward normals (nx_i is for the face opposite node i)
!
          nx1 = 0.5*((y2 - y4)*(z3 - z4) - (y3 - y4)*(z2 - z4))
          ny1 = 0.5*((z2 - z4)*(x3 - x4) - (z3 - z4)*(x2 - x4))
          nz1 = 0.5*((x2 - x4)*(y3 - y4) - (x3 - x4)*(y2 - y4))
!
          nx2 = 0.5*((y3 - y4)*(z1 - z4) - (y1 - y4)*(z3 - z4))
          ny2 = 0.5*((z3 - z4)*(x1 - x4) - (z1 - z4)*(x3 - x4))
          nz2 = 0.5*((x3 - x4)*(y1 - y4) - (x1 - x4)*(y3 - y4))
!
          nx3 = 0.5*((y1 - y4)*(z2 - z4) - (y2 - y4)*(z1 - z4))
          ny3 = 0.5*((z1 - z4)*(x2 - x4) - (z2 - z4)*(x1 - x4))
          nz3 = 0.5*((x1 - x4)*(y2 - y4) - (x2 - x4)*(y1 - y4))
!
          nx4 = -nx1 -nx2 -nx3
          ny4 = -ny1 -ny2 -ny3
          nz4 = -nz1 -nz2 -nz3
!
! Compute cell volume
!
          vol = (((y2-y1)*(z3-z1) - (y3-y1)*(z2-z1))*(x4-x1)
     &          -((x2-x1)*(z3-z1) - (x3-x1)*(z2-z1))*(y4-y1)
     &          +((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))*(z4-z1))/6.
!
 
               rho1  = qnode(1,node1)
               u1    = qnode(2,node1)/rho1
               v1    = qnode(3,node1)/rho1
               w1    = qnode(4,node1)/rho1
               p1    = gm1*(qnode(5,node1) 
     &               - .5*rho1*(u1*u1 + v1*v1 + w1*w1))
               rho2  = qnode(1,node2)
               u2    = qnode(2,node2)/rho2
               v2    = qnode(3,node2)/rho2
               w2    = qnode(4,node2)/rho2
               p2    = gm1*(qnode(5,node2) 
     &               - .5*rho2*(u2*u2 + v2*v2 + w2*w2))
               rho3  = qnode(1,node3)
               u3    = qnode(2,node3)/rho3
               v3    = qnode(3,node3)/rho3
               w3    = qnode(4,node3)/rho3
               p3    = gm1*(qnode(5,node3) 
     &               - .5*rho3*(u3*u3 + v3*v3 + w3*w3))
               rho4  = qnode(1,node4)
               u4    = qnode(2,node4)/rho4
               v4    = qnode(3,node4)/rho4
               w4    = qnode(4,node4)/rho4
               p4    = gm1*(qnode(5,node4) 
     &               - .5*rho4*(u4*u4 + v4*v4 + w4*w4))
!
!  Compute viscosity for the cell
!
          T1 = gamma*p1/rho1
          T2 = gamma*p2/rho2
          T3 = gamma*p3/rho3
          T4 = gamma*p4/rho4
!
          rmu1 = (1. + cstar)/(T1 + cstar)*T1**1.5 + amut(node1)
          rmu2 = (1. + cstar)/(T2 + cstar)*T2**1.5 + amut(node2)
          rmu3 = (1. + cstar)/(T3 + cstar)*T3**1.5 + amut(node3)
          rmu4 = (1. + cstar)/(T4 + cstar)*T4**1.5 + amut(node4)
!
          rmu = 0.25*(rmu1 + rmu2 + rmu3 + rmu4)
!
!  Now form gradients of velocity
!
          const = -1.0/(3.0*vol)
!
          ux = const*((u1-u4)*nx1 + (u2-u4)*nx2 + (u3-u4)*nx3)
          uy = const*((u1-u4)*ny1 + (u2-u4)*ny2 + (u3-u4)*ny3)
          uz = const*((u1-u4)*nz1 + (u2-u4)*nz2 + (u3-u4)*nz3)
!
          vx = const*((v1-v4)*nx1 + (v2-v4)*nx2 + (v3-v4)*nx3)
          vy = const*((v1-v4)*ny1 + (v2-v4)*ny2 + (v3-v4)*ny3)
          vz = const*((v1-v4)*nz1 + (v2-v4)*nz2 + (v3-v4)*nz3)
!
          wx = const*((w1-w4)*nx1 + (w2-w4)*nx2 + (w3-w4)*nx3)
          wy = const*((w1-w4)*ny1 + (w2-w4)*ny2 + (w3-w4)*ny3)
          wz = const*((w1-w4)*nz1 + (w2-w4)*nz2 + (w3-w4)*nz3)
!
          xnorm = nx4
          ynorm = ny4
          znorm = nz4
!
!  Now compute components of stress vector acting on the face
!
          termx = xmr*rmu*(xnorm*(c43*ux - c23*(vy + wz))
     &                    +ynorm*(uy + vx)
     &                    +znorm*(uz + wx))
!
          termy = xmr*rmu*(xnorm*(uy + vx)
     &                    +ynorm*(c43*vy - c23*(ux + wz))
     &                    +znorm*(vz + wy))
!
          termz = xmr*rmu*(xnorm*(uz + wx)
     &                    +ynorm*(vz + wy)
     &                    +znorm*(c43*wz - c23*(ux + vy)))
!
!  Now dot the stress vector acting on the surface face with 
!  a unit vector in the drag (lift) direction.  This is the
!  magnitude of the friction force acting on the face in the
!  drag (lift) direction
!
!  Find unit vectors in drag and lift directions
!
          nxd =   cos(alpha/conv) * cos(yaw/conv)
          nyd = - sin(yaw/conv)
          nzd =   sin(alpha/conv) * cos(yaw/conv)
!
          nxl = - sin(alpha/conv)
          nyl =   0.
          nzl =   cos(alpha/conv)
!
!  Now do the dot product to get the force in the drag (lift) direction
!
!  I think the signs are right on the following two equations, but I
!  wouldn't stake my life on it.  They've got to do with the force
!  being on the body or on the fluid.  The way they are set right now
!  gives the logical results (increase in Cd, decrease in Cl).
!
          forced = - (termx*nxd + termy*nyd + termz*nzd)
          forcel = - (termx*nxl + termy*nyl + termz*nzl)
!
!  Now add things
!
          cdv = cdv + forced
          clv = clv + forcel
        endif
!
      enddo
!
      return
      end


!================================ DELTAT2 ============================72
!
! Calculate a time step for each cell
! Note that this routine assumes conservative variables
!
!=====================================================================72
      subroutine DELTAT2(nnodes,nedge,qvec,cdt,
     &                  x,y,z,vol,xn,yn,zn,rl,evec,
     &                  sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,
     &                  nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
     &                  irank,nvertices)
!
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb

      integer ierr
      PetscLogDouble flops
      real x(1),y(1),z(1),vol(1)
      real xn(1),yn(1),zn(1),rl(1)
      real sxn(1),syn(1),szn(1)
      real vxn(1),vyn(1),vzn(1)
      real fxn(1),fyn(1),fzn(1)

      integer isnode(1),ivnode(1),ifnode(1)

      dimension cdt(1)
c  
#if defined(INTERLACING)
      real qvec(5,nvertices)
      integer evec(2,nedge)
#define qnode(i,j) qvec(i,j)
#define eptr(j,i) evec(i,j)
#else
      real qvec(nvertices,5)
      integer evec(nedge,2)
#define qnode(i,j) qvec(j,i)
#define eptr(i,j) evec(i,j)
#endif

c  If local time steping, loop over faces
c  and calculate time step as cdt = V/(sum(|u.n| +c.area)
c  This is time step for cfl=1. We will multiply by cfl number later
c
c First loop over nodes and zero out cdt
c
      flops = 0.0
      if (dt.lt.0.0) then
         do 1000 i = 1,nvertices
            cdt(i) = 0.0
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
c
c xnorm = x-normal x area of face  
c ynorm = y-normal x area of face  
c znorm = z-normal x area of face  
c
               rho1 = qnode(1,node1)
               u1   = qnode(2,node1)/rho1
               v1   = qnode(3,node1)/rho1
               w1   = qnode(4,node1)/rho1
               e1   = qnode(5,node1)
               p1   = gm1*(e1 - 0.5*rho1*(u1*u1 + v1*v1 + w1*w1))
               c1   = sqrt(gamma*p1/rho1)
c
               rho2 = qnode(1,node2)
               u2   = qnode(2,node2)/rho2
               v2   = qnode(3,node2)/rho2
               w2   = qnode(4,node2)/rho2
               e2   = qnode(5,node2)
               p2   = gm1*(e2 - 0.5*rho2*(u2*u2 + v2*v2 + w2*w2))
               c2   = sqrt(gamma*p2/rho2)
c
c Get average values on face 
c
               u    = 0.5*(u1 + u2)
               v    = 0.5*(v1 + v2)
               w    = 0.5*(w1 + w2)
               c    = 0.5*(c1 + c2)
c
               term = abs(u*xnorm + v*ynorm + w*znorm) + c*area
               cdt(node1) = cdt(node1) + term
               cdt(node2) = cdt(node2) + term
c
 1020    continue
c
         flops = flops + 51.0*nedge
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
c
            rho = qnode(1,inode)
            u   = qnode(2,inode)/rho
            v   = qnode(3,inode)/rho
            w   = qnode(4,inode)/rho
            e   = qnode(5,inode)
            p   = gm1*(e - 0.5*rho*(u*u + v*v + w*w))
            c   = sqrt(gamma*p/rho)
c
            Vn = abs(xnorm*u + ynorm*v + znorm*w) + c*area
            cdt(inode) = cdt(inode) + Vn
c
 1030    continue
c
         flops = flops + 30.0*nsnode
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
            rho = qnode(1,inode)
            u   = qnode(2,inode)/rho
            v   = qnode(3,inode)/rho
            w   = qnode(4,inode)/rho
            e   = qnode(5,inode)
            p   = gm1*(e - 0.5*rho*(u*u + v*v + w*w))
            c   = sqrt(gamma*p/rho)
c
            Vn = abs(xnorm*u + ynorm*v + znorm*w) + c*area
            cdt(inode) = cdt(inode) + Vn
c
 1040    continue
c
         flops = flops + 30.0*nvnode
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
c
            rho = qnode(1,inode)
            u   = qnode(2,inode)/rho
            v   = qnode(3,inode)/rho
            w   = qnode(4,inode)/rho
            e   = qnode(5,inode)
            p   = gm1*(e - 0.5*rho*(u*u + v*v + w*w))
            c   = sqrt(gamma*p/rho)
c
            Vn = abs(xnorm*u + ynorm*v + znorm*w) + c*area
            cdt(inode) = cdt(inode) + Vn
 1050    continue
c
         flops = flops + 30.0*nfnode
 
c
c Now cdt has sum(|u.n| + c*area)
c
         do 1060 n = 1,nvertices
c           if (cdt(n) .eq. 0.0) then
c              write(*,*) 'cdt is zero for vertex # ',n
c              stop
c           endif
            cdt(n) = vol(n)/cdt(n)
 1060    continue
         flops = flops + nvertices
      else
c
c If not doing local time stepping just set cdt=1
c
         do 1070 n = 1,nvertices
            cdt(n) = 1.0
 1070    continue
      end if
      call PetscLogFlops(flops,ierr)
c     timeNorm = 0.0
c     qNorm = 0.0
c     do i = 1, nvertices
c        timeNorm = timeNorm + cdt(i)*cdt(i)
c        do j = 1, 5
c           qnorm = qnorm + qnode(j,i)*qnode(j,i)
c        enddo
c     enddo
c     print *, 'time norm is ', sqrt(timeNorm)
c     print *, 'qnorm is ', sqrt(qNorm)
c
c End of subroutine DELTAT2
c
      return
      end


c================================= SPLIT =============================72
c
c Calculates the fluxes on the face and performs the flux balance
c Note that this subroutine uses primative variables 
c
c=====================================================================72
      subroutine SPLIT(nnodes, ncell, nedge,
     &                nsface, nvface, nfface, isface, ivface, ifface,
     &                nsnode, nvnode, nfnode, isnode, ivnode, ifnode,
     &                nnfacet,f2ntn,nnbound,
     &                nvfacet,f2ntv,nvbound,
     &                nffacet,f2ntf,nfbound,
     &                grad,
     &                evec, qvec,
     &                resvec,
     &                x, y, z,
     &                xn, yn, zn, ra,
     &                sxn, syn, szn,
     &                vxn, vyn, vzn,
     &                fxn, fyn, fzn, phiv, irank,nvertices)

c
      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo,ncyc
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0

      PetscLogDouble flops
      integer isface(1),ivface(1),ifface(1)
      integer isnode(1),ivnode(1),ifnode(1)
      integer f2ntn(nnfacet,4)
      integer f2ntv(nvfacet,4)
      integer f2ntf(nffacet,4)

      real sxn(1),syn(1),szn(1)
      real vxn(1),vyn(1),vzn(1)
      real fxn(1),fyn(1),fzn(1)
      real x(nvertices), y(nvertices), z(nvertices)
c     real phi(nvertices,5)
      real xn(nedge), yn(nedge), zn(nedge), ra(nedge)


#if defined(INTERLACING)
      real qvec(5,nvertices) 
      real grad(3,5,nvertices) 
      real resvec(5,nnodes)
      real phiv(5,nvertices)
      integer evec(2,nedge)
#define qnode(i,j) qvec(i,j)
#define res(i,j) resvec(i,j)
#define gradx(x,y) grad(1,x,y)
#define grady(x,y) grad(2,x,y)
#define gradz(x,y) grad(3,x,y)
#define phi(i,j) phiv(i,j)
#define eptr(j,i) evec(i,j)
#else
      real qvec(nvertices,5) 
      real grad(nvertices,5,3) 
      real resvec(nnodes,5)
      real phiv(nvertices,5)
      integer evec(nedge,2)
#define qnode(i,j) qvec(j,i)
#define res(i,j) resvec(j,i)
#define gradx(x,y) grad(y,x,1)
#define grady(x,y) grad(y,x,2)
#define gradz(x,y) grad(y,x,3)
#define phi(i,j) phiv(j,i)
#define eptr(i,j) evec(i,j)
#endif

c Loop over all the faces and calculate a flux 
c
       flops = 0.0
       second = 1.0
       if(ntt.le.nitfo)second = 0.0
c      print *, 'Second is' , second
       do 1010 n = 1, nedge
        node1 = eptr(n,1)
        node2 = eptr(n,2)
        if ((node1 .le. nnodes).or.(node2 .le. nnodes)) then
c
c Get fluxes on "left" side of face
c
          xmean = .5*(x(node1) + x(node2))
          ymean = .5*(y(node1) + y(node2))
          zmean = .5*(z(node1) + z(node2))
c
          xnorm  = xn(n)
          ynorm  = yn(n)
          znorm  = zn(n)
          area   = ra(n)
c
          rx = second*(xmean - x(node1))
          ry = second*(ymean - y(node1))
          rz = second*(zmean - z(node1))
c
          rho   = qnode(1,node1) + phi(1,node1)*gradx(1,node1)*rx
     1                           + phi(1,node1)*grady(1,node1)*ry
     2                           + phi(1,node1)*gradz(1,node1)*rz
c
          u     = qnode(2,node1) + phi(2,node1)*gradx(2,node1)*rx
     1                           + phi(2,node1)*grady(2,node1)*ry
     2                           + phi(2,node1)*gradz(2,node1)*rz
c
          v     = qnode(3,node1) + phi(3,node1)*gradx(3,node1)*rx
     1                           + phi(3,node1)*grady(3,node1)*ry
     2                           + phi(3,node1)*gradz(3,node1)*rz

          w     = qnode(4,node1) + phi(4,node1)*gradx(4,node1)*rx
     1                           + phi(4,node1)*grady(4,node1)*ry
     2                           + phi(4,node1)*gradz(4,node1)*rz

          press = qnode(5,node1) + phi(5,node1)*gradx(5,node1)*rx
     1                           + phi(5,node1)*grady(5,node1)*ry
     2                           + phi(5,node1)*gradz(5,node1)*rz
c
          q2    = u*u + v*v + w*w
          enrgy = press/gm1 + 0.5*rho*q2
          c     = sqrt(gamma*press/rho)
          ubar  = xnorm*u + ynorm*v + znorm*w
          fmach = ubar/c 
          ubp2a = -ubar + 2.*c
	  flops = flops + 61.0
c
          if(abs(fmach).lt.1.0)then
           fluxp1 = area*0.25*rho*c*(fmach + 1)**2
           fluxp2 = fluxp1*(xnorm*ubp2a/gamma + u)
           fluxp3 = fluxp1*(ynorm*ubp2a/gamma + v)
           fluxp4 = fluxp1*(znorm*ubp2a/gamma + w)
           fluxp5 = fluxp1*((-gm1*ubar*ubar 
     &                  + 2.*gm1*ubar*c + 2.*c*c)/(gamma*gamma - 1)
     &                  + 0.5*q2)
           flops = flops + 34.0
          else if(fmach.ge.1.0)then
           fluxp1 = area*rho*ubar
           fluxp2 = area*(rho*u*ubar + xnorm*press)
           fluxp3 = area*(rho*v*ubar + ynorm*press)
           fluxp4 = area*(rho*w*ubar + znorm*press)
           fluxp5 = area*(enrgy + press)*ubar
           flops = flops + 20.0
          else
           fluxp1 = 0.0
           fluxp2 = 0.0
           fluxp3 = 0.0
           fluxp4 = 0.0
           fluxp5 = 0.0
          end if
c
c Get fluxes on "right" side of face
c
          rx = second*(xmean - x(node2))
          ry = second*(ymean - y(node2))
          rz = second*(zmean - z(node2))
c
          rho   = qnode(1,node2) + phi(1,node2)*gradx(1,node2)*rx
     1                           + phi(1,node2)*grady(1,node2)*ry
     2                           + phi(1,node2)*gradz(1,node2)*rz
c
          u     = qnode(2,node2) + phi(2,node2)*gradx(2,node2)*rx
     1                           + phi(2,node2)*grady(2,node2)*ry
     2                           + phi(2,node2)*gradz(2,node2)*rz
c
          v     = qnode(3,node2) + phi(3,node2)*gradx(3,node2)*rx
     1                           + phi(3,node2)*grady(3,node2)*ry
     2                           + phi(3,node2)*gradz(3,node2)*rz

          w     = qnode(4,node2) + phi(4,node2)*gradx(4,node2)*rx
     1                           + phi(4,node2)*grady(4,node2)*ry
     2                           + phi(4,node2)*gradz(4,node2)*rz

          press = qnode(5,node2) + phi(5,node2)*gradx(5,node2)*rx
     1                           + phi(5,node2)*grady(5,node2)*ry
     2                           + phi(5,node2)*gradz(5,node2)*rz
c
          q2    = u*u + v*v + w*w
          enrgy = press/gm1 + 0.5*rho*q2
          c     = sqrt(gamma*press/rho)
          ubar  = xnorm*u + ynorm*v + znorm*w
          fmach = ubar/c 
          ubm2a = -ubar - 2.*c
	  flops = flops + 55.0
c
          if(abs(fmach).lt.1.0)then
           fluxm1 = -area*0.25*rho*c*(fmach - 1)**2
           fluxm2 = fluxm1*(xnorm*ubm2a/gamma + u)
           fluxm3 = fluxm1*(ynorm*ubm2a/gamma + v)
           fluxm4 = fluxm1*(znorm*ubm2a/gamma + w)
           fluxm5 = fluxm1*((-gm1*ubar*ubar 
     &                  - 2.*gm1*ubar*c + 2.*c*c)/(gamma*gamma - 1)
     &                  + 0.5*q2)
           flops = flops + 35.0
c
          else if(fmach.le.-1.0)then
           fluxm1 = area*rho*ubar
           fluxm2 = area*(rho*u*ubar + xnorm*press)
           fluxm3 = area*(rho*v*ubar + ynorm*press)
           fluxm4 = area*(rho*w*ubar + znorm*press)
           fluxm5 = area*(enrgy + press)*ubar
           flops = flops + 20.0
          else
           fluxm1 = 0.0
           fluxm2 = 0.0
           fluxm3 = 0.0
           fluxm4 = 0.0
           fluxm5 = 0.0
          end if
c
          if (node1 .le. nnodes) then
           res(1,node1) = res(1,node1) + (fluxp1 + fluxm1)
           res(2,node1) = res(2,node1) + (fluxp2 + fluxm2)
           res(3,node1) = res(3,node1) + (fluxp3 + fluxm3)
           res(4,node1) = res(4,node1) + (fluxp4 + fluxm4)
           res(5,node1) = res(5,node1) + (fluxp5 + fluxm5)
           flops = flops + 5.0
          endif
c
          if (node2 .le. nnodes) then
           res(1,node2) = res(1,node2) - (fluxp1 + fluxm1)
           res(2,node2) = res(2,node2) - (fluxp2 + fluxm2)
           res(3,node2) = res(3,node2) - (fluxp3 + fluxm3)
           res(4,node2) = res(4,node2) - (fluxp4 + fluxm4)
           res(5,node2) = res(5,node2) - (fluxp5 + fluxm5)
           flops = flops + 5.0
          endif
        endif
c
 1010 continue
  
c     c68 = 6./8.
c     c18 = 1./8.
      c68 = 0.75
      c18 = 0.125
c
c Loop over the boundaries 
c First do inviscid faces
c
      do 2020 n = 1, nnfacet
               node1 = isnode(f2ntn(n,1))
               node2 = isnode(f2ntn(n,2))
               node3 = isnode(f2ntn(n,3)) 

               x1 = x(node1)
               y1 = y(node1)
               z1 = z(node1)
               p1 = qnode(5,node1)

               x2 = x(node2)
               y2 = y(node2)
               z2 = z(node2)
               p2 = qnode(5,node2)

               x3 = x(node3)
               y3 = y(node3)
               z3 = z(node3)
               p3 = qnode(5,node3)
               
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
               xnorm =-0.5*(ay*bz - az*by)/3.
               ynorm = 0.5*(ax*bz - az*bx)/3.
               znorm =-0.5*(ax*by - ay*bx)/3.

               pa = c68*p1 + c18*(p2 + p3)
               pb = c68*p2 + c18*(p3 + p1)
               pc = c68*p3 + c18*(p1 + p2)
c
               flops = flops + 35.0
               if (node1 .le. nnodes) then
                res(2,node1) = res(2,node1) + xnorm*pa
                res(3,node1) = res(3,node1) + ynorm*pa
                res(4,node1) = res(4,node1) + znorm*pa
	        flops = flops + 6.0
               endif

               if (node2 .le. nnodes) then
                res(2,node2) = res(2,node2) + xnorm*pb
                res(3,node2) = res(3,node2) + ynorm*pb
                res(4,node2) = res(4,node2) + znorm*pb
	        flops = flops + 6.0
               endif

               if (node3 .le. nnodes) then
                res(2,node3) = res(2,node3) + xnorm*pc
                res(3,node3) = res(3,node3) + ynorm*pc
                res(4,node3) = res(4,node3) + znorm*pc
	        flops = flops + 6.0
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
               p1 = qnode(5,node1)

               x2 = x(node2)
               y2 = y(node2)
               z2 = z(node2)
               p2 = qnode(5,node2)

               x3 = x(node3)
               y3 = y(node3)
               z3 = z(node3)
               p3 = qnode(5,node3)
               
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
               xnorm =-0.5*(ay*bz - az*by)/3.
               ynorm = 0.5*(ax*bz - az*bx)/3.
               znorm =-0.5*(ax*by - ay*bx)/3.

               pa = c68*p1 + c18*(p2 + p3)
               pb = c68*p2 + c18*(p3 + p1)
               pc = c68*p3 + c18*(p1 + p2)
c
	       flops = flops + 35.0
c
               if (node1 .le. nnodes) then
                res(2,node1) = res(2,node1) + xnorm*pa
                res(3,node1) = res(3,node1) + ynorm*pa
                res(4,node1) = res(4,node1) + znorm*pa
                flops = flops + 6.0

               endif

               if (node2 .le. nnodes) then
                res(2,node2) = res(2,node2) + xnorm*pb
                res(3,node2) = res(3,node2) + ynorm*pb
                res(4,node2) = res(4,node2) + znorm*pb
                flops = flops + 6.0
               endif

               if (node3 .le. nnodes) then
                res(2,node3) = res(2,node3) + xnorm*pc
                res(3,node3) = res(3,node3) + ynorm*pc
                res(4,node3) = res(4,node3) + znorm*pc
                flops = flops + 6.0
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
c       p = qnode(5,inode)
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
c       p = qnode(5,inode)
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
c
c Now do far-field
c
      s0 = c0*c0/(gamma*rho0**gm1)
      xgm1 = 1.0/gm1
      uout = u0
      vout = v0
      wout = w0
      cout = c0
      flops = flops + 5.0
       do 1040 i = 1,nfnode
         inode   = ifnode(i)
c
c Calculate R+ and R-
c Then get the normal velocity and the
c speed of sound on the boundary
c
         xnorm   = fxn(i)
         ynorm   = fyn(i)
         znorm   = fzn(i)
         area    = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
         xnorm   = xnorm/area
         ynorm   = ynorm/area
         znorm   = znorm/area
         rhoi    = qnode(1,inode)
         ui      = qnode(2,inode)
         vi      = qnode(3,inode)
         wi      = qnode(4,inode)
         unormi  = ui*xnorm + vi*ynorm + wi*znorm
         unormo  = uout*xnorm + vout*ynorm + wout*znorm
         u2      = ui*ui + vi*vi + wi*wi
         a2      = gamma*qnode(5,inode)/rhoi
         a       = sqrt(a2)
         ai      = a
         rplus   = unormi + 2.0*a/gm1
         rminus  = unormo - 2.0*cout/gm1
         flops = flops + 32.0
         if (unormi .gt. 1.0) then 
            rminus = unormi - 2.0*a/gm1
            flops = flops + 3.0
         endif
         if (unormi .lt. -1.0) then
            rplus = unormo + 2.0*cout/gm1
            flops = flops + 3.0
         endif
         unorm   = 0.5*(rplus + rminus)
         a       = 0.25*gm1*(rplus - rminus)
         flops   = flops + 5.0
c
c If unorm > 0 this is outflow: take variables from inside
c If unorm < 0 this is inflow:  take variables from outside
c
         if(unorm.gt.0.0)then
          u = qnode(2,inode) + xnorm*(unorm - unormi)
          v = qnode(3,inode) + ynorm*(unorm - unormi)
          w = qnode(4,inode) + znorm*(unorm - unormi)
          s = ai*ai/(gamma*rhoi**gm1)
          flops = flops + 13.0
         else
          u = uout + xnorm*(unorm - unormo)
          v = vout + ynorm*(unorm - unormo)
          w = wout + znorm*(unorm - unormo)
          s = s0
          flops = flops + 9.0
         end if
          rho  = (a*a/(gamma*s))**xgm1
          p    = rho*a*a/gamma
          e    = p/gm1 + 0.5*rho*(u*u + v*v + w*w)
          ubar = xnorm*u + ynorm*v + znorm*w
          flops = flops + 21.0
c
         if (inode .le. nnodes) then
          res(1,inode) = res(1,inode)+area*rho*ubar
          res(2,inode) = res(2,inode)+area*(rho*u*ubar+xnorm*p)
          res(3,inode) = res(3,inode)+area*(rho*v*ubar+ynorm*p)
          res(4,inode) = res(4,inode)+area*(rho*w*ubar+znorm*p)
          res(5,inode) = res(5,inode)+area*(e + p)*ubar
          flops = flops + 25.0
         endif
c
 1040  continue
       call PetscLogFlops(flops,ierr)
c
c End of subroutine SPLIT
c
      return
      end

!================================= ROE ===============================72
!
! This routine computes the fluxes using Roe's approximate Riemann 
! solver.
! Remember that for this subroutine, q(5) is the pressure
!
!=====================================================================72
      subroutine ROE(nnodes, ncell, nedge,
     &                nsface, nvface, nfface, isface, ivface, ifface,
     &                nsnode, nvnode, nfnode, isnode, ivnode, ifnode,
     &                nnfacet,f2ntn,nnbound,
     &                nvfacet,f2ntv,nvbound,
     &                nffacet,f2ntf,nfbound,
     &                grad,
     &                evec, qvec,
     &                resvec,
     &                x, y, z,
     &                xn, yn, zn, ra,
     &                sxn, syn, szn,
     &                vxn, vyn, vzn,
     &                fxn, fyn, fzn, phiv,irank,nvertices)

c
      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo,ncyc
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      integer ierr
      PetscLogDouble flops
      integer isface(1),ivface(1),ifface(1)
      integer isnode(1),ivnode(1),ifnode(1)
      integer f2ntn(nnfacet,4)
      integer f2ntv(nvfacet,4)
      integer f2ntf(nffacet,4)

      real sxn(1),syn(1),szn(1)
      real vxn(1),vyn(1),vzn(1)
      real fxn(1),fyn(1),fzn(1)
      real x(nvertices), y(nvertices), z(nvertices)
      real xn(nedge), yn(nedge), zn(nedge), ra(nedge)


#if defined(INTERLACING)
      real qvec(5,nvertices) 
      real grad(3,5,nvertices) 
      real resvec(5,nnodes)
      real phiv(5,nvertices)
      integer evec(2,nedge)
#define qnode(i,j) qvec(i,j)
#define res(i,j) resvec(i,j)
#define gradx(x,y) grad(1,x,y)
#define grady(x,y) grad(2,x,y)
#define gradz(x,y) grad(3,x,y)
#define phi(i,j) phiv(i,j)
#define eptr(j,i) evec(i,j)
#else
      real qvec(nvertices,5) 
      real grad(nvertices,5,3) 
      real resvec(nnodes,5)
      real phiv(nvertices,5)
      integer evec(nedge,2)
#define qnode(i,j) qvec(j,i)
#define res(i,j) resvec(j,i)
#define gradx(x,y) grad(y,x,1)
#define grady(x,y) grad(y,x,2)
#define gradz(x,y) grad(y,x,3)
#define phi(i,j) phiv(j,i)
#define eptr(i,j) evec(i,j)
#endif
c
c Loop over all the faces and calculate a flux 
c
       flops = 0.0
       if (ntt.le.nitfo) then
        second = 0.0
       else
        second = 1.0
       endif
c      print *, 'Second is' , second
       do 1010 n = 1, nedge
        node1 = eptr(n,1)
        node2 = eptr(n,2)
        if ((node1 .le. nnodes).or.(node2 .le. nnodes)) then
c
c Calculate unit normal to face and length of face
c
          xmean = .5*(x(node1) + x(node2))
          ymean = .5*(y(node1) + y(node2))
          zmean = .5*(z(node1) + z(node2))
          xnorm  = xn(n)
          ynorm  = yn(n)
          znorm  = zn(n)
          area   = ra(n)
          flops = flops + 6.0
c
c Get variables on "left" side of face
c
          if (second .eq. 0) then
            rhol   = qnode(1,node1)
            ul     = qnode(2,node1) 
            vl     = qnode(3,node1)
            wl     = qnode(4,node1) 
            q2l    = ul*ul + vl*vl + wl*wl
            pressl = qnode(5,node1) 
            enrgyl = pressl/gm1 + .5*rhol*q2l
            Hl     = (enrgyl + pressl)/rhol
            cl     = sqrt(gamma*pressl/rhol)
            ubarl  = xnorm*ul + ynorm*vl + znorm*wl
c
c Get variables on "right" side of face
c
            rhor   = qnode(1,node2)
            ur     = qnode(2,node2) 
            vr     = qnode(3,node2) 
            wr     = qnode(4,node2) 
            q2r    = ur*ur + vr*vr + wr*wr
            pressr = qnode(5,node2)  
            enrgyr = pressr/gm1 + .5*rhor*q2r
            Hr     = (enrgyr + pressr)/rhor
            cr     = sqrt(gamma*pressr/rhor)
            ubarr  = xnorm*ur + ynorm*vr + znorm*wr
            flops  = flops + 38.0
          else
            rx     = xmean - x(node1)
            ry     = ymean - y(node1)
            rz     = zmean - z(node1)
            rhol   = qnode(1,node1) + phi(1,node1)*gradx(1,node1)*rx 
     &                              + phi(1,node1)*grady(1,node1)*ry
     &                              + phi(1,node1)*gradz(1,node1)*rz
            ul     = qnode(2,node1) + phi(2,node1)*gradx(2,node1)*rx 
     &                              + phi(2,node1)*grady(2,node1)*ry 
     &                              + phi(2,node1)*gradz(2,node1)*rz 
            vl     = qnode(3,node1) + phi(3,node1)*gradx(3,node1)*rx 
     &                              + phi(3,node1)*grady(3,node1)*ry 
     &                              + phi(3,node1)*gradz(3,node1)*rz 
            wl     = qnode(4,node1) + phi(4,node1)*gradx(4,node1)*rx 
     &                              + phi(4,node1)*grady(4,node1)*ry 
     &                              + phi(4,node1)*gradz(4,node1)*rz 
            q2l    = ul*ul + vl*vl + wl*wl
            pressl = qnode(5,node1) + phi(5,node1)*gradx(5,node1)*rx 
     &                              + phi(5,node1)*grady(5,node1)*ry 
     &                              + phi(5,node1)*gradz(5,node1)*rz 
            enrgyl = pressl/gm1 + .5*rhol*q2l
            Hl     = (enrgyl + pressl)/rhol
            cl     = sqrt(gamma*pressl/rhol)
            ubarl  = xnorm*ul + ynorm*vl + znorm*wl
c
c Get variables on "right" side of face
c
            rx     = xmean - x(node2)
            ry     = ymean - y(node2)
            rz     = zmean - z(node2)
            rhor   = qnode(1,node2) + phi(1,node2)*gradx(1,node2)*rx 
     &                              + phi(1,node2)*grady(1,node2)*ry
     &                              + phi(1,node2)*gradz(1,node2)*rz
            ur     = qnode(2,node2) + phi(2,node2)*gradx(2,node2)*rx 
     &                              + phi(2,node2)*grady(2,node2)*ry 
     &                              + phi(2,node2)*gradz(2,node2)*rz 
            vr     = qnode(3,node2) + phi(3,node2)*gradx(3,node2)*rx 
     &                              + phi(3,node2)*grady(3,node2)*ry 
     &                              + phi(3,node2)*gradz(3,node2)*rz 
            wr     = qnode(4,node2) + phi(4,node2)*gradx(4,node2)*rx 
     &                              + phi(4,node2)*grady(4,node2)*ry 
     &                              + phi(4,node2)*gradz(4,node2)*rz 
            q2r    = ur*ur + vr*vr + wr*wr
            pressr = qnode(5,node2) + phi(5,node2)*gradx(5,node2)*rx 
     &                              + phi(5,node2)*grady(5,node2)*ry 
     &                              + phi(5,node2)*gradz(5,node2)*rz 
            enrgyr = pressr/gm1 + .5*rhor*q2r
            Hr     = (enrgyr + pressr)/rhor
            cr     = sqrt(gamma*pressr/rhor)
            ubarr  = xnorm*ur + ynorm*vr + znorm*wr
            flops  = flops + 134.0
          endif
c
c Compute rho averages
c
            rho = sqrt(rhol*rhor)
c           wat = sqrt(rhol)/(sqrt(rhol) + sqrt(rhor))
            wat = rho/(rho + rhor)
            u   = ul*wat + ur*(1. - wat)
            v   = vl*wat + vr*(1. - wat)
            w   = wl*wat + wr*(1. - wat)
            H   = Hl*wat + Hr*(1. - wat)
            q2  = u*u + v*v + w*w
            c   = sqrt(gm1*(H - 0.5*q2))
            ubar = xnorm*u + ynorm*v + znorm*w
c
c Now compute eigenvalues, eigenvectors, and strengths
c
            eig1 = abs(ubar + c)
            eig2 = abs(ubar - c)
            eig3 = abs(ubar)
 
            drho   = rhor - rhol
            dpress = pressr - pressl
            du     = ur - ul
            dv     = vr - vl
            dw     = wr - wl
            dubar  = ubarr - ubarl
 
            c2 = c*c
c
c jumps have units of density
c
            dv1 = 0.5*(dpress + rho*c*dubar)/c2
            dv2 = 0.5*(dpress - rho*c*dubar)/c2
            dv3 = rho
            dv4 = (c*c*drho - dpress)/c2
 
            r21 = u + c*xnorm
            r31 = v + c*ynorm
            r41 = w + c*znorm
            r51 = H + c*ubar

            r22 = u - c*xnorm
            r32 = v - c*ynorm
            r42 = w - c*znorm
            r52 = H - c*ubar

            r23 = du - dubar*xnorm
            r33 = dv - dubar*ynorm
            r43 = dw - dubar*znorm
            r53 = u*du + v*dv + w*dw - ubar*dubar
         
            r24 = u
            r34 = v
            r44 = w
            r54 = 0.5*q2
 
            t1 = eig1*dv1     + eig2*dv2
     &                        + eig3*dv4
            t2 = eig1*r21*dv1 + eig2*r22*dv2 
     &         + eig3*r23*dv3 + eig3*r24*dv4
            t3 = eig1*r31*dv1 + eig2*r32*dv2 
     &         + eig3*r33*dv3 + eig3*r34*dv4
            t4 = eig1*r41*dv1 + eig2*r42*dv2 
     &         + eig3*r43*dv3 + eig3*r44*dv4
            t5 = eig1*r51*dv1 + eig2*r52*dv2 
     &         + eig3*r53*dv3 + eig3*r54*dv4
c
c Compute flux using variables from left side of face 
c
            fluxp1 = area*rhol*ubarl
            fluxp2 = area*(rhol*ul*ubarl + xnorm*pressl)
            fluxp3 = area*(rhol*vl*ubarl + ynorm*pressl)
            fluxp4 = area*(rhol*wl*ubarl + znorm*pressl)
            fluxp5 = area*(enrgyl + pressl)*ubarl
c
c Now the right side
c
            fluxm1 = area*rhor*ubarr
            fluxm2 = area*(rhor*ur*ubarr + xnorm*pressr)
            fluxm3 = area*(rhor*vr*ubarr + ynorm*pressr)
            fluxm4 = area*(rhor*wr*ubarr + znorm*pressr)
            fluxm5 = area*(enrgyr + pressr)*ubarr
 
            res1 = 0.5*(fluxp1 + fluxm1 - area*t1)
            res2 = 0.5*(fluxp2 + fluxm2 - area*t2)
            res3 = 0.5*(fluxp3 + fluxm3 - area*t3)
            res4 = 0.5*(fluxp4 + fluxm4 - area*t4)
            res5 = 0.5*(fluxp5 + fluxm5 - area*t5)
            flops = flops + 177.0
            if (node1 .le. nnodes) then
             res(1,node1) = res(1,node1) + res1
             res(2,node1) = res(2,node1) + res2
             res(3,node1) = res(3,node1) + res3
             res(4,node1) = res(4,node1) + res4
             res(5,node1) = res(5,node1) + res5
             flops = flops + 5.0
            endif
c
            if (node2 .le. nnodes) then
             res(1,node2) = res(1,node2) - res1
             res(2,node2) = res(2,node2) - res2
             res(3,node2) = res(3,node2) - res3
             res(4,node2) = res(4,node2) - res4
             res(5,node2) = res(5,node2) - res5
             flops = flops + 5.0
            endif
          endif
c
 1010 continue
c     c68 = 6./8.
c     c18 = 1./8.
      c68 = 0.75
      c18 = 0.125
c
c Loop over the boundaries 
c First do inviscid faces
c
c  Switch the constants for now
c
      c68 = 1.
      c18 = 0.
c
      do 2020 n = 1, nnfacet
               node1 = isnode(f2ntn(n,1))
               node2 = isnode(f2ntn(n,2))
               node3 = isnode(f2ntn(n,3)) 

               x1 = x(node1)
               y1 = y(node1)
               z1 = z(node1)
               p1 = qnode(5,node1)

               x2 = x(node2)
               y2 = y(node2)
               z2 = z(node2)
               p2 = qnode(5,node2)

               x3 = x(node3)
               y3 = y(node3)
               z3 = z(node3)
               p3 = qnode(5,node3)
               
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
               xnorm =-0.5*(ay*bz - az*by)/3.
               ynorm = 0.5*(ax*bz - az*bx)/3.
               znorm =-0.5*(ax*by - ay*bx)/3.

               pa = c68*p1 + c18*(p2 + p3)
               pb = c68*p2 + c18*(p1 + p3)
               pc = c68*p3 + c18*(p1 + p2)
c
               flops = flops + 35.0
               if (node1 .le. nnodes) then
                res(2,node1) = res(2,node1) + xnorm*pa
                res(3,node1) = res(3,node1) + ynorm*pa
                res(4,node1) = res(4,node1) + znorm*pa
	        flops = flops + 6.0
               endif

               if (node2 .le. nnodes) then
                res(2,node2) = res(2,node2) + xnorm*pb
                res(3,node2) = res(3,node2) + ynorm*pb
                res(4,node2) = res(4,node2) + znorm*pb
	        flops = flops + 6.0
               endif

               if (node3 .le. nnodes) then
                res(2,node3) = res(2,node3) + xnorm*pc
                res(3,node3) = res(3,node3) + ynorm*pc
                res(4,node3) = res(4,node3) + znorm*pc
	        flops = flops + 6.0
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
               p1 = qnode(5,node1)

               x2 = x(node2)
               y2 = y(node2)
               z2 = z(node2)
               p2 = qnode(5,node2)

               x3 = x(node3)
               y3 = y(node3)
               z3 = z(node3)
               p3 = qnode(5,node3)
               
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
               xnorm =-0.5*(ay*bz - az*by)/3.
               ynorm = 0.5*(ax*bz - az*bx)/3.
               znorm =-0.5*(ax*by - ay*bx)/3.

               pa = c68*p1 + c18*(p2 + p3)
               pb = c68*p2 + c18*(p1 + p3)
               pc = c68*p3 + c18*(p1 + p2)
c
	       flops = flops + 35.0
c
               if (node1 .le. nnodes) then
                res(2,node1) = res(2,node1) + xnorm*pa
                res(3,node1) = res(3,node1) + ynorm*pa
                res(4,node1) = res(4,node1) + znorm*pa
                flops = flops + 6.0

               endif

               if (node2 .le. nnodes) then
                res(2,node2) = res(2,node2) + xnorm*pb
                res(3,node2) = res(3,node2) + ynorm*pb
                res(4,node2) = res(4,node2) + znorm*pb
                flops = flops + 6.0
               endif

               if (node3 .le. nnodes) then
                res(2,node3) = res(2,node3) + xnorm*pc
                res(3,node3) = res(3,node3) + ynorm*pc
                res(4,node3) = res(4,node3) + znorm*pc
                flops = flops + 6.0
               endif

 3020 continue
c
c The next section of code is for when you dont care about
c preserving linear data on boundary. Also, doing this 
c matches the left hand side when not doing Newton-Krylov
c Usually just go around unless you are just experimenting
c
c      goto 1025
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
c       p = qnode(5,inode)
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
c       p = qnode(5,inode)
c
c       if (inode .le. nnodes) then
c        res(2,inode) = res(2,inode) + xnorm*p
c        res(3,inode) = res(3,inode) + ynorm*p
c        res(4,inode) = res(4,inode) + znorm*p
c       endif
c
c1030 continue
c
c 1025 continue
c
c
c Now do far-field
c
      s0 = c0*c0/(gamma*rho0**gm1)
      xgm1 = 1.0/gm1
      uout = u0
      vout = v0
      wout = w0
      cout = c0
      flops = flops + 5.0
      do 1040 i = 1,nfnode
         inode   = ifnode(i)
c
c Calculate R+ and R-
c Then get the normal velocity and the
c speed of sound on the boundary
c
         xnorm   = fxn(i)
         ynorm   = fyn(i)
         znorm   = fzn(i)
         area    = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
         xnorm   = xnorm/area
         ynorm   = ynorm/area
         znorm   = znorm/area
         rhoi    = qnode(1,inode)
         ui      = qnode(2,inode)
         vi      = qnode(3,inode)
         wi      = qnode(4,inode)
         unormi  = ui*xnorm + vi*ynorm + wi*znorm
         unormo  = uout*xnorm + vout*ynorm + wout*znorm
         u2      = ui*ui + vi*vi + wi*wi
         a2      = gamma*qnode(5,inode)/rhoi
         a       = sqrt(a2)
         ai      = a
         rplus   = unormi + 2.0*a/gm1
         rminus  = unormo - 2.0*cout/gm1
         flops = flops + 32.0
         if (unormi .gt. 1.0) then 
            rminus = unormi - 2.0*a/gm1
            flops = flops + 3.0
         endif
         if (unormi .lt. -1.0) then
            rplus = unormo + 2.0*cout/gm1
            flops = flops + 3.0
         endif
         unorm   = 0.5*(rplus + rminus)
         a       = 0.25*gm1*(rplus - rminus)
         flops   = flops + 5.0
c
c If unorm > 0 this is outflow: take variables from inside
c If unorm < 0 this is inflow:  take variables from outside
c
         if(unorm.gt.0.0)then
          u = qnode(2,inode) + xnorm*(unorm - unormi)
          v = qnode(3,inode) + ynorm*(unorm - unormi)
          w = qnode(4,inode) + znorm*(unorm - unormi)
          s = ai*ai/(gamma*rhoi**gm1)
          flops = flops + 13.0
         else
          u = uout + xnorm*(unorm - unormo)
          v = vout + ynorm*(unorm - unormo)
          w = wout + znorm*(unorm - unormo)
          s = s0
          flops = flops + 9.0
         end if
         rho  = (a*a/(gamma*s))**xgm1
         p    = rho*a*a/gamma
         e    = p/gm1 + 0.5*rho*(u*u + v*v + w*w)
         ubar = xnorm*u + ynorm*v + znorm*w
         flops = flops + 21.0
c
         if (inode .le. nnodes) then
          res(1,inode) = res(1,inode)+area*rho*ubar
          res(2,inode) = res(2,inode)+area*(rho*u*ubar+xnorm*p)
          res(3,inode) = res(3,inode)+area*(rho*v*ubar+ynorm*p)
          res(4,inode) = res(4,inode)+area*(rho*w*ubar+znorm*p)
          res(5,inode) = res(5,inode)+area*(e + p)*ubar
          flops = flops + 25.0
         endif
c
 1040  continue
       call PetscLogFlops(flops,ierr)
c
c End of subroutine ROE
c
      return
      end
c
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
      subroutine SUMGS(nnodes,nedge,evec,x,y,z,
     1                 rxy,
     2                 irank,nvertices)

      real x(nvertices),y(nvertices),z(nvertices)
      integer ierr
c      real rxy(7,nnodes)
c
#if defined(INTERLACING)
      real rxy(7,nnodes)
      integer evec(2,nedge)
#define r11(x) rxy(1,x)
#define r12(x) rxy(2,x)
#define r13(x) rxy(3,x)
#define r22(x) rxy(4,x)
#define r23(x) rxy(5,x)
#define r33(x) rxy(6,x)
#define r44(x) rxy(7,x)
#define eptr(j,i) evec(i,j)
#else
      real rxy(nnodes,7)
      integer evec(nedge,2)
#define r11(x) rxy(x,1)
#define r12(x) rxy(x,2)
#define r13(x) rxy(x,3)
#define r22(x) rxy(x,4)
#define r23(x) rxy(x,5)
#define r33(x) rxy(x,6)
#define r44(x) rxy(x,7)
#define eptr(i,j) evec(i,j)
#endif
      PetscSizeT sizeofscalar
!
!
!    Initialize all the rij to 0.0 
!
!     do 1000 i = 1,nnodes
!        r11(i) = 0.0
!        r12(i) = 0.0
!        r13(i) = 0.0
!        r22(i) = 0.0
!        r23(i) = 0.0
!        r33(i) = 0.0
!        r44(i) = 0.0
!1000 continue
      call PetscDataTypeGetSize(PETSC_SCALAR,sizeofscalar,ierr)
      call PetscMemzero(rxy,sizeofscalar*7*nnodes,ierr)
!
! Now loop over the edges and accumulate the r's 
!
      do 1020 n = 1, nedge
!
          node1 = eptr(n,1)
          node2 = eptr(n,2)
!
          x1 = x(node1)
          y1 = y(node1)
          z1 = z(node1)
          x2 = x(node2)
          y2 = y(node2)
          z2 = z(node2)
!
          dx = x2 - x1
          dy = y2 - y1
          dz = z2 - z1
          dist = sqrt(dx*dx + dy*dy + dz*dz)
          weight = 1.0/dist
          weight = 1.0
          w2 = weight*weight
!
          if (node1 .le. nnodes) then
           r11(node1) = r11(node1) + (x2 - x1)*(x2 - x1)*w2
           r12(node1) = r12(node1) + (x2 - x1)*(y2 - y1)*w2
           r13(node1) = r13(node1) + (x2 - x1)*(z2 - z1)*w2
          endif
!
          if (node2 .le. nnodes) then
           r11(node2) = r11(node2) + (x1 - x2)*(x1 - x2)*w2
           r12(node2) = r12(node2) + (x1 - x2)*(y1 - y2)*w2
           r13(node2) = r13(node2) + (x1 - x2)*(z1 - z2)*w2
          endif
 1020 continue

!
! Now calculate ||x|| = r11(1) by taking the square root
! Also divide r12(1) and r13(1) by ||x||
!
      do 1030 i = 1,nnodes
        r11(i) = sqrt(r11(i))
        r12(i) = r12(i)/r11(i)
        r13(i) = r13(i)/r11(i)
 1030 continue
!
! Now calculate r22(0) and r23(0)
!
      do 1050 n = 1, nedge
!
          node1 = eptr(n,1)
          node2 = eptr(n,2)
!
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
          weight = 1.0/dist
          weight = 1.0
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
c 'Now finish getting r22 and r23'
c */
      do 1060 i = 1,nnodes
        r22(i) = sqrt(r22(i))
        r23(i) = r23(i)/r22(i)
 1060 continue
c /*
c 'Now all we have to do is get r33'
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
          weight = 1.0/dist
          weight = 1.0
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
c
c /*
c 'Now just get the magnitude of r33'
c */
      do 1090 i = 1,nnodes
        r33(i) = sqrt(r33(i))
 1090 continue
c
c Added by Dinesh Kaushik (6/27/97)
c /*'The following addition changes the meaning of r11 .. r33. r44
c is the new parameter introduced by me. The new definitions
c are taken from LSTGS (where these parameters 
c are used finally).
c r11->w11
c r22->w22
c r33->w33
c r12->r12r11
c r13->r13r11
c r23->r23r22
c r44->rmess' */

      do i = 1,nnodes
           w11 = 1./(r11(i)*r11(i))
           w22 = 1./(r22(i)*r22(i))
           w33 = 1./(r33(i)*r33(i))
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
c
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
      subroutine LSTGS(nnodes,nedge,evec,
     1                 qvec,grad,x,y,z,
     2                 rxy,irank,nvertices)
c
c      real rxy(7,nnodes)
c      real grad(3,4,nnodes)
c      real qnode(4,nvertices)
      real x(nvertices),y(nvertices),z(nvertices)
c
      integer ierr
      PetscLogDouble flops
C     logging variables
      integer grad_event, flag
      integer node1_event, node2_event
      character * 16 grad_label, node1_label, node2_label
      data flag/-1/,grad_label/'GRAD            '/
      data node1_label/'NODE1           '/
      data node2_label/'NODE2           '/
      save grad_event, grad_label, flag
      save node1_event,node2_event,node1_label, node2_label


c
      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo,ncyc
c
#if defined(INTERLACING)
      real qvec(5,nvertices)
      real rxy(7,nnodes)
      real grad(3,5,nnodes)
      integer evec(2,nedge)
#define qnode(i,j) qvec(i,j)
#define gradx(x,y) grad(1,x,y)
#define grady(x,y) grad(2,x,y)
#define gradz(x,y) grad(3,x,y)
#define r11(x) rxy(1,x)
#define r12(x) rxy(2,x)
#define r13(x) rxy(3,x)
#define r22(x) rxy(4,x)
#define r23(x) rxy(5,x)
#define r33(x) rxy(6,x)
#define r44(x) rxy(7,x)
#define eptr(j,i) evec(i,j)
#else
      real qvec(nvertices,5)
      real rxy(nnodes,7)
      real grad(nnodes,5,3)
      integer evec(nedge,2)
#define qnode(i,j) qvec(j,i)
#define gradx(x,y) grad(y,x,1)
#define grady(x,y) grad(y,x,2)
#define gradz(x,y) grad(y,x,3)
#define r11(x) rxy(x,1)
#define r12(x) rxy(x,2)
#define r13(x) rxy(x,3)
#define r22(x) rxy(x,4)
#define r23(x) rxy(x,5)
#define r33(x) rxy(x,6)
#define r44(x) rxy(x,7)
#define eptr(i,j) evec(i,j)
#endif
      PetscSizeT sizeofscalar

c     if (flag .eq. -1) then
c        call PLogEventRegister(grad_event,grad_label,'blue:',ierr)
c        call PLogEventRegister(node1_event,node1_label,'red:',ierr)
c        call PLogEventRegister(node2_event,node2_label,'green:',ierr)
c        flag = 1
c     endif
c     call PLogEventBegin(grad_event,0,0,0,0,ierr)

      flops = 0.0
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
      call PetscDataTypeGetSize(PETSC_SCALAR,sizeofscalar,ierr)
      call PetscMemzero(grad,sizeofscalar*15*nnodes,ierr)
c
c Convert the energy to total enthalpy or pressure
c
      if (ihane.eq.1) then
         call ETOH(nvertices,qvec)
      else
         call ETOP(nvertices,qvec)
      end if
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
           weight = 1.0
c          w2 = weight*weight
c
           dx1 = weight*dx1
           dy1 = weight*dy1
           dz1 = weight*dz1

           flops = flops + 6.0
c
c         call PLogEventBegin(node1_event,0,0,0,0,ierr)
          if (node1 .le. nnodes) then
           dq1 = weight*(qnode(1,node2) - qnode(1,node1))
           dq2 = weight*(qnode(2,node2) - qnode(2,node1))
           dq3 = weight*(qnode(3,node2) - qnode(3,node1))
           dq4 = weight*(qnode(4,node2) - qnode(4,node1))
           dq5 = weight*(qnode(5,node2) - qnode(5,node1))
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
           gradx(5,node1) = gradx(5,node1) + termx*dq5
           grady(5,node1) = grady(5,node1) + termy*dq5
           gradz(5,node1) = gradz(5,node1) + termz*dq5
c
           flops = flops + 49.0
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
           dq5 = weight*(qnode(5,node1) - qnode(5,node2))
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
           gradx(5,node2) = gradx(5,node2) + termx*dq5
           grady(5,node2) = grady(5,node2) + termy*dq5
           gradz(5,node2) = gradz(5,node2) + termz*dq5
c
           flops = flops + 52.0
          endif
c         call PLogEventEnd(node2_event,0,0,0,0,ierr)
c       endif
c
 1020  continue
c
c Convert total enthalpy or pressure to energy
c
      if (ihane.eq.1) then
         call HTOE(nvertices,qvec)
      else
         call PTOE(nvertices,qvec)
      end if
c     end if
      call PetscLogFlops(flops,ierr)
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
      subroutine GETRES(nnodes,ncell,nedge,nsface,nvface,nfface,nbface,
     1                  nsnode,nvnode,nfnode,isface,ivface,ifface,
     2                  ileast,isnode,ivnode,ifnode,
     &                  nnfacet,f2ntn,nnbound,
     &                  nvfacet,f2ntv,nvbound,
     &                  nffacet,f2ntf,nfbound,
     &                  evec,
     3                  sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,
     4                  xn,yn,zn,rl,qvec,cdt,x,y,z,area,
     5                  grad,
     5                  resvec,
     6                  turbre,slen,c2n,c2e,
     7                  us,vs,as,phiv,
     &                  amut,ires,
     &                  irank, nvertices)
c
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,
     &             nstage,ncyct,iramp,nitfo,ncyc
      common/history/rms(3000),clw(3000),cdw(3000),cmxw(3000),
     2               cmyw(3000),cmzw(3000),cxw(3000),cyw(3000),
     3               czw(3000),xres(3000)
      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
c
      integer evec(nedge,2)
      integer isface(1),ivface(1),ifface(1)
      integer isnode(1),ivnode(1),ifnode(1)
      integer c2n(ncell,4),c2e(ncell,6)
      integer f2ntn(nnfacet,4)
      integer f2ntv(nvfacet,4)
      integer f2ntf(nffacet,4)
c
      real us(nbface,3,5),vs(nbface,3,5),as(nbface,3,5)
      real sxn(1),syn(1),szn(1)
      real vxn(1),vyn(1),vzn(1)
      real fxn(1),fyn(1),fzn(1)
      real x(nvertices),y(nvertices),z(nvertices),area(nvertices)
      real xn(1),yn(1),zn(1),rl(1)
      real turbre(1),slen(1)
      real qvec(5,nvertices)
      real cdt(nvertices)
      real grad(3,5,nvertices)
c     real dq(nnodes,4)
c     real r11(nvertices),r12(nvertices),r13(nvertices)
c     real r22(nvertices),r23(nvertices),r33(nvertices)
      real amut(nnodes)
c
C     logging variable
      integer flux_event, delta2_event, flag
      character * 16 flux_label, delta2_label
      data flag/-1/,flux_label/'FLUX            '/
      data delta2_label/'DELTA2          '/
      save flux_event,delta2_event,flag, flux_label,delta2_label
c
#if defined(INTERLACING)
       real resvec(5,nnodes)
       real phiv(5,nvertices)
#define res(i,j) resvec(i,j)
#define phi(i,j) phiv(i,j)
#else
       real resvec(nnodes,5)
       real phiv(nvertices,5)
#define res(i,j) resvec(j,i)
#define phi(i,j) phiv(j,i)
#endif
c

c     if (flag .eq. -1) then
c        call PLogEventRegister(delta2_event,delta2_label,'red:',ierr)
c        call PLogEventRegister(flux_event,flux_label,'blue:',ierr)         
c        flag = 1
c     endif
c
c  /*'Set boundary conditions
c
c---------------------------
c Dont call boundary conditions so we just have weak enforcement
c Actually all this does is to zero the velocity for viscous walls
c since this wasn't done in INIT
c
c     call BC(nnodes,nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
c    1        qnode,sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn)
c
c     if (ivisc.eq.2.or.ivisc.eq.4.or.ivisc.eq.6)then
c        write (6,*) "BTERMS not implemented yet"
c        stop
c        call BTERMS(nnodes,ncell,nedge,nsface,nvface,nfface,nbface,
c    1               isface,ivface,ifface,qnode,evec,us,vs,as,x,y,c2n)
c     end if
c--------------------------
c'*/
c Calculate the time step
c
      if(ires.eq.1) goto 888
c     call PLogEventBegin(delta2_event,0,0,0,0,ierr)
      call DELTAT2(nnodes,nedge,qvec,cdt,
     &             x,y,z,area,xn,yn,zn,rl,evec,
     &             sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,
     &             nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
     &             irank,nvertices)
c     call PLogEventEnd(delta2_event,0,0,0,0,ierr)
  888 continue
c
c Convert the energy to total enthalpy or pressure
c
      if (ihane.eq.1) then
         call ETOH(nvertices,qvec)
      else
         call ETOP(nvertices,qvec)
      end if
c /*'
c   Calculate the gradients 
c   ----Kyle seems to recommend only LSTGS for gradients,
c   so I have commented the GETGRAD call - DKK (1/17/97)
c
c     if (ileast.eq.0) then
c        call GETGRAD(nnodes,ncell,nedge,nsface,nvface,nfface,
c    &                isface,ivface,ifface,evec,ncolor,ncount,
c    &                qnode,gradx,grady,x,y,
c    &                area,wx,wy,xn,yn,rl)
c
c     else if (ileast.eq.4) then
c     if (ileast.eq.4) then
c        call LSTGS(nnodes,nedge,evec,
c    1              qnode,gradx,grady,gradz,x,y,z,
c    2              r11,r12,r13,r22,r23,r33,irank,nvertices)
c     end if
c
c zero out residuals (viscous residuals are zeroed in vfluxnew)
c '*/
      do 1002 i = 1,nnodes
          res(1,i)=0.0
          res(2,i)=0.0
          res(3,i)=0.0
          res(4,i)=0.0
          res(5,i)=0.0
 1002 continue
      do i = 1,nvertices
          phi(1,i)=1.0
          phi(2,i)=1.0
          phi(3,i)=1.0
          phi(4,i)=1.0
          phi(5,i)=1.0
      enddo
c
c /*'
c If not doing Newton-Krylov and iflim=1 call the Flux Limiter
c
c     if(iflim.eq.1.and.ifcn.ne.1)then
c     if (iflim.ne.2) then
c      call TIMLIM(nnodes,nedge,qnode,res,dq,phiv,ncolor,ncount,
c    &             gradx,grady,gradz,x,y,z,eptr)
c 
c If we used the limiter we need to zero out the residual again
c since we used it for scratch space
c
c     do 1003 i = 1,nnodes
c         res(1,i)=0.0
c         res(2,i)=0.0
c         res(3,i)=0.0
c         res(4,i)=0.0
c         res(5,i)=0.0
c1003   continue
c
c     end if
c
c   Split the fluxes and perform the flux balance
c
c       call PLogEventBegin(flux_event,0,0,0,0,ierr)

c        call FLUX(nnodes,ncell,nedge,
c    &            nsface,nvface,nfface,isface,ivface,ifface,
c    &            nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
c    &            nnfacet,f2ntn,nnbound,
c    &            nvfacet,f2ntv,nvbound,
c    &            nffacet,f2ntf,nfbound,
c    &            grad,evec,qvec,
c    &            x,y,z,resvec,xn,yn,zn,rl,sxn,syn,szn,vxn,vyn,vzn,
c    &            fxn,fyn,fzn,phiv,irank,nvertices)
c '*/
      if (ihane.eq.1)then
         write (6,*) "SPLITH not implemented"
         stop
c
c /*'
c        call splith(nnodes,ncell,nedge,nsface,nvface,nfface,
c    &               isface,ivface,ifface,grad,evec,ncolor,
c    &               ncount,qvec,x,y,resvec,xn,yn,rl,
c    &               nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
c    &               sxn,syn,vxn,vyn,fxn,fyn,phiv)
c '*/
      else if(ihane.eq.0)then
         call SPLIT(nnodes,ncell,nedge,nsface,nvface,nfface,
     &             isface,ivface,ifface,nsnode,nvnode,nfnode,
     &             isnode,ivnode,ifnode,
     &             nnfacet,f2ntn,nnbound,
     &             nvfacet,f2ntv,nvbound,
     &             nffacet,f2ntf,nfbound,
     &             grad,
     &             evec,qvec,
     &             resvec,
     &             x,y,z,
     &             xn,yn,zn,rl,
     &             sxn,syn,szn,vxn,vyn,vzn,
     &             fxn,fyn,fzn,phiv,irank,nvertices)
      else if(ihane.eq.3)then
         write (6,*) "SPLITS not implemented"
         stop
c
c /*'
c        call splits(nnodes,ncell,nedge,nsface,nvface,nfface,
c    &              isface,ivface,ifface,grad,evec,ncolor,
c    &              ncount,qvec,x,y,resvec,xn,yn,rl,
c    &              nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
c    &              sxn,syn,vxn,vyn,fxn,fyn,phiv)
c '*/
      else if(ihane.eq.4)then
         write (6,*) "AUSM not implemented"
         stop
c
c /*'
c        call ausm(nnodes,ncell,nedge,nsface,nvface,nfface,
c    &             isface,ivface,ifface,grad,evec,ncolor,
c    &             ncount,qvec,x,y,resvec,xn,yn,rl,
c    &             nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
c    &             sxn,syn,vxn,vyn,fxn,fyn,phiv)
c '*/
      else 
         call ROE(nnodes,ncell,nedge,
     &            nsface,nvface,nfface,isface,ivface,ifface,
     &            nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
     &            nnfacet,f2ntn,nnbound,
     &            nvfacet,f2ntv,nvbound,
     &            nffacet,f2ntf,nfbound,
     &            grad,
     &            evec,qvec,
     &            resvec,
     &            x,y,z,
     &            xn,yn,zn,rl,
     &            sxn,syn,szn,
     &            vxn,vyn,vzn,
     &            fxn,fyn,fzn,phiv,irank,nvertices)
c
c /*'
c Use ROEIO for internal flows
c
c        call ROEIO(nnodes,ncell,nedge,
c    &              nsface,nvface,nfface,isface,ivface,ifface,
c    &              nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
c    &              nnfacet,f2ntn,nnbound,
c    &              nvfacet,f2ntv,nvbound,
c    &              nffacet,f2ntf,nfbound,
c    &              grad,evec,ncolor,ncount,qvec,
c    &              x,y,z,resvec,xn,yn,zn,rl,sxn,syn,szn,vxn,vyn,vzn,
c    &              fxn,fyn,fzn,phiv,irank,nvertices)
c '*/
      end if
c
c /*'
c        call PLogEventEnd(flux_event,0,0,0,0,ierr)
c
c calculate viscous fluxes
c
c     if (ivisc.gt.0) then
c        if (ihane.eq.1) then
c           write (6,*) "VFLUXPH2 not implemented"
c           stop
c        else
c           call VISRHS (nnodes,ncell,nedge,
c           call EDGEVIS(nnodes,ncell,nedge,
c    &                  nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
c    &                  nsface,nvface,nfface,isface,ivface,ifface,
c    &                  nnfacet,f2ntn,nnbound,ncolorn,countn,
c    &                  nvfacet,f2ntv,nvbound,ncolorv,countv,
c    &                  nffacet,f2ntf,nfbound,ncolorf,countf,
c    &                  nccolor,nccount,
c    &                  evec,c2n,c2e,
c    &                  sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,
c    &                  x,y,z,gradx,grady,gradz,
c    &                  qnode,amut,res,phiv)
c     end if
c '*/
c Convert total enthalpy or pressure to energy
c
      if (ihane.eq.1) then
         call HTOE(nvertices,qvec)
      else
         call PTOE(nvertices,qvec)
      end if
c
c End of subroutine GETRES
c
      return
      end
c
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
      SUBROUTINE OPENM(irank)
c
c  TAPE7  -- input:  mach number, angle of attack etc.. 
c  TAPE9  -- input:  reads restart file
c  TAPE10 -- output: residual history
c  TAPE11 -- output: writes restart file
c  TAPE12 -- output: writes residual and lift for plotting
c  TAPE13 -- output: writes flowfield for contour plotting
c
      OPEN(UNIT= 7,FILE='ginput.faces',
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
      subroutine GETIA(nnodes,nedge,evec,ia,ideg,irank)
      integer ia(1),ideg(1)
#if defined(INTERLACING)
       integer evec(2,nedge)
#define eptr(j,i) evec(i,j)
#else
       integer evec(nedge,2)
#define eptr(i,j) evec(i,j)
#endif
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
c
c===================================================================
c
c Get the IA, JA, and IAU arrays
c
c===================================================================
      subroutine GETJA(nnodes,nedge,evec,ia,ja,iwork,irank)
c     integer ia(1),ja(1),iau(1),iwork(1)
      integer ia(1),ja(1),iwork(1)
#if defined(INTERLACING)
       integer evec(2,nedge)
#define eptr(j,i) evec(i,j)
#else
       integer evec(nedge,2)
#define eptr(i,j) evec(i,j)
#endif
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
      close(90)
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
      subroutine SORTER(istart,iend,ja,inode)
c     integer ja(1),iau(1)
      integer ja(1)
c
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
      subroutine IREAD(unit,n,iper,arr)
c
c===================================================================
      integer unit, n, iper
      integer arr(n,iper)
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
      subroutine RREAD(unit,n,iper,arr)
c
c===================================================================
      integer unit, n, iper
      real arr(n,iper)
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
