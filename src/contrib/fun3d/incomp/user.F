!---------------------------------------------------------------
! The following subroutines are from node2t.f in the original
! code - D. K. Kaushik (1/17/97)
!---------------------------------------------------------------
!
!
! 2-D Navier Stokes on Unstructured Grids

!============================== FORLINK ==============================72
!
!  FORLINK establishes links between FORTRAN common blocks and C
!
!=====================================================================72
      subroutine FORLINK()

      implicit none
#include <petsc/finclude/petscsys.h>
      PetscScalar  title(20),beta,alpha
      PetscScalar  Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      PetscScalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,                        &
     &        nstage,ncyct,iramp,nitfo
      PetscScalar  gtol
      integer icycle,nsrch,ilu0,ifcn
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,                 &
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,         &
     &             nstage,ncyct,iramp,nitfo
      common/gmcom/gtol,icycle,nsrch,ilu0,ifcn

      call CLINK(title,cfl1,gtol)
!
! End of subroutine FORLINK
!
      return
      end

!============================== Block_Initialization  ======== =======72
!
! Initializes the common blocks members for turbulence model
!
!=====================================================================72
      block data Block_Initialization
      implicit none
#include <petsc/finclude/petscsys.h>
      PetscScalar vkar,cmu,ce1,ce2
      PetscScalar aplus1,aplus2,turbinf
      PetscScalar cb1,sig,cb2,cw1,cw2
      PetscScalar cw3,cv1,ct1,ct2,ct3,ct4
      common/turb/vkar,cmu,ce1,ce2,aplus1,aplus2,turbinf
      common/spalrt/cb1,sig,cb2,cw1,cw2,cw3,cv1,ct1,ct2,ct3,ct4
      data vkar,cmu,ce1,ce2/0.41,0.09,1.2,2.0 /
      data aplus1,aplus2,turbinf/26.0,10.0,0.1 /
      data cb1,sig,cb2,cw2,cw3/0.1355,0.66667,0.622,0.3,2.0/
!
! Comment out old coefficients
!      data cv1,ct1,ct2,ct3,ct4/7.1,1.0,2.0,1.1,2.0/
!
       data cv1,ct1,ct2,ct3,ct4/7.1d0,1.0d0,2.0d0,1.2d0,0.5d0/

      end

!================================== INIT =============================72
!
! Initializes the flow field
!
!=====================================================================72
      subroutine INIT(nnodes, qvec, turbre, amut,nvnode, ivnode,irank)
      implicit none
#include <petsc/finclude/petscsys.h>
      PetscScalar  turbre(1),amut(1)
      integer ivnode(1),nnodes,nvnode,irank
      PetscScalar  title(20),beta,alpha
      PetscScalar  Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      PetscScalar  gamma,gm1,gp1,gm1g,gp1g,ggm1
      PetscScalar  p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      PetscScalar  vkar,cmu,ce1,ce2
      PetscScalar  aplus1,aplus2,turbinf
      PetscScalar  cb1,sig,cb2,cw1,cw2,cw3
      PetscScalar  cv1,ct1,ct2,ct3,ct4
      PetscScalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,                          &
     &        nstage,ncyct,iramp,nitfo
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,                   &
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/turb/vkar,cmu,ce1,ce2,aplus1,aplus2,turbinf
      common/spalrt/cb1,sig,cb2,cw1,cw2,cw3,cv1,ct1,ct2,ct3,ct4
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,           &
     &             nstage,ncyct,iramp,nitfo
      PetscScalar  pi,conv,rmu,chi,fv1
      Integer i,k,n
#if defined(INTERLACING)
       PetscScalar qvec(4,nnodes)
#define qnod(i,j) qvec(i,j)
#else
       PetscScalar qvec(nnodes,4)
#define qnod(i,j) qvec(j,i)
#endif
!
      cw1 = cb1/vkar/vkar + (1.0d0 + cb2)/sig
      if (ivisc.gt.4d0)turbinf = 1.341946d0
!      if (ivisc.gt.4)turbinf = 0.5
      if (ivisc.gt.4.and.itran.eq.1)turbinf = 0.1d0
!
! Note that for Spalarts model, I use turbinf as the freestream value of
! the dependent variable just as in the Baldwin-Barth model.
! The constant is set so that in the freestream, nu_t=0.009 (1.341946)
!
!     print *, "I am in INIT"
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
         qnod(1,n) = p0
         qnod(2,n) = u0
         qnod(3,n) = v0
         qnod(4,n) = w0
      enddo
      if (ivisc.eq.3) then
       do n = 1,nnodes
         turbre(n)  = 0.0d0
         amut(n)    = 0.0d0
       enddo
      endif
!
! If viscous, zero out the velocity on the surface
!
!     print *, "Just Before Viscous"

      do 9010 i = 1,nvnode
!
! Compute the velocity normal to the surface
!
        k       = ivnode(i)
        qnod(2,k) = 0.0d0
        qnod(3,k) = 0.0d0
        qnod(4,k) = 0.0d0
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
            rmu = 1.0d0
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

!================================ READR1 ====================================
!
!  Reads input parameters
!
!============================================================================
      subroutine READR1(ileast, irank)

      implicit none
#include <petsc/finclude/petscsys.h>
      PetscScalar  title(20),beta,alpha
      PetscScalar  Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      PetscScalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,                                &
     &        nstage,ncyct,iramp,nitfo
      PetscScalar  gtol
      integer icycle,nsrch,ilu0,ifcn
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,                         &
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,                 &
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
       if (ivisc.eq.5.or.ivisc.eq.6) write(10,123)itran,nbtran
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

!================================ TECFLO =============================72
!
!  Writes a formatted file that contains an input file for TECPLOT
!
!=====================================================================72
      subroutine TECFLO(nnodes,                                            &
     &                  nnbound,nvbound,nfbound,                           &
     &                  nnfacet,nvfacet,nffacet,                           &
     &                  nsnode,nvnode,nfnode,                              &
     &                  title,x,y,z,q,                                     &
     &                  nnpts,nntet,nvpts,nvtet,nfpts,nftet,               &
     &                  f2ntn,f2ntv,f2ntf,isnode,ivnode,ifnode,            &
     &                  irank)
!
      implicit none
#include <petsc/finclude/petscsys.h>
      PetscScalar pinf
      PetscScalar gamma,gm1,gp1,gm1g,gp1g,ggm1
      PetscScalar p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      integer nnodes,nnbound,nvbound,nfbound,                              &
     &        nnfacet,nvfacet,nffacet,                                     &
     &        nsnode,nvnode,nfnode,irank
      integer nntet(1),nnpts(1)
      integer nvtet(1),nvpts(1)
      integer nftet(1),nfpts(1)
      integer f2ntn(nnfacet,4),f2ntv(nvfacet,4),f2ntf(nffacet,4)
      integer isnode(1),ivnode(1),ifnode(1)
      PetscScalar  x(1),y(1),z(1),q(4,nnodes)
      integer i,j,i1,i2,i3,ic,isp,ist

      character c4*4,title*20

      pinf = p0
!
!   + write tecplot header
!
      write(13,'(a)') 'TITLE="' // title // '"'
      write(13,'(a)') 'VARIABLES="X     ","Y     ","Z     ","RHO   ",',     &
     &     '"U     ","V     ","W     ","P/Pinf"'
!
!   + do a zone-title, so we can keep track of things
!   + start with solid-wall boundary surfaces
!
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
!
!   + start with viscous boundary surfaces
!
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
!
!   + do the far-field boundary surfaces
!
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

!
!      End of subroutine TECFLO
!
 1000  format('ZONE T="',a,'", Z=0, I=',i6,', J=',i6,', F=FEBLOCK')
 1010  format(1P10E13.5)
 1020  format(4I10)

       return
       end

!================================ FORCE  =============================72
!
!  Calculates the forces
!  Modified - D. K. Kaushik (1/15/97)
!  Added new parameters - clift, cdrag, cmom, irank, nvertices
!
!=====================================================================72
      subroutine FORCE(nnodes,nedge,                                    &
     &                 isnode,ivnode,                                   &
     &                 nnfacet,f2ntn,nnbound,                           &
     &                 nvfacet,f2ntv,nvbound,                           &
     &                 evec,qvec,xyz,sface_bit,vface_bit,               &
     &                 clift, cdrag, cmom,irank,nvertices)
      implicit none
#include <petsc/finclude/petscsys.h>
      PetscScalar  title(20),beta,alpha
      PetscScalar  Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      PetscScalar gamma,gm1,gp1,gm1g,gp1g,ggm1
      PetscScalar p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      PetscScalar rms(1001),clw(1001),cdw(1001),                             &
     &       cmw(1001),xres(1001)
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,                 &
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/history/rms,clw,cdw,cmw,xres

      integer nnodes,nedge,nnfacet,nvfacet,                             &
     &        nnbound,nvbound,                                          &
     &        irank,nvertices
      integer isnode(1),ivnode(1)
      integer f2ntn(nnfacet,4)
      integer f2ntv(nvfacet,4)
      integer sface_bit(nnfacet), vface_bit(nvfacet)

      PetscScalar clift, cdrag, cmom
      PetscScalar clloc,cdloc,cmloc
      PetscScalar clglo,cdglo,cmglo
      PetscScalar cl_ghost, cd_ghost, cm_ghost
      PetscScalar coef_loc(3), coef_glo(3)
      PetscScalar pi,conv,csa,sna,xr,yr,zr
      PetscScalar x1,y1,z1,x2,y2,z2,x3,y3
      PetscScalar z3,xmid,ymid,zmid
      PetscScalar p1,p2,p3,u1,u2,u3,v1,v2,v3
      PetscScalar w1,w2,w3,ax,ay,az,bx,by,bz
      PetscScalar xnorm,ynorm,znorm
      PetscScalar dcx,dcy,dcz,cp,press
      integer icount,n,node1,node2,node3,ierr

#if defined(INTERLACING)
      PetscScalar qvec(4,nvertices)
      PetscScalar xyz(3,nvertices)
      integer evec(2,nedge)
#define qnod(i,j) qvec(i,j)
#define x(i) xyz(1,i)
#define y(i) xyz(2,i)
#define z(i) xyz(3,i)
#define eptr(j,i) evec(i,j)
#else
      PetscScalar qvec(nvertices,4)
      PetscScalar xyz(nvertices,3)
      integer evec(nedge,2)
#define qnod(i,j) qvec(j,i)
#define x(i) xyz(i,1)
#define y(i) xyz(i,2)
#define z(i) xyz(i,3)
#define eptr(i,j) evec(i,j)
#endif
!
      pi = 4.d0*atan(1.d0)
      conv = 180.d0/pi
      csa=cos(alpha/conv)
      sna=sin(alpha/conv)
!
!  initialize forces to zero
!
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
!
!  norm points outward, away from grid interior.
!  norm magnitude is area of surface triangle.
!
               xnorm =-0.5d0*(ay*bz - az*by)
               ynorm = 0.5d0*(ax*bz - az*bx)
               znorm =-0.5d0*(ax*by - ay*bx)

               p1 = qnod(1,node1)
               u1 = qnod(2,node1)
               v1 = qnod(3,node1)
               w1 = qnod(4,node1)

               p2 = qnod(1,node2)
               u2 = qnod(2,node2)
               v2 = qnod(3,node2)
               w2 = qnod(4,node2)

               p3 = qnod(1,node3)
               u3 = qnod(2,node3)
               v3 = qnod(3,node3)
               w3 = qnod(4,node3)

               press = (p1 + p2 + p3)/3.0d0
               cp    = 2.d0*(press - 1.0d0)
!
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
                cmloc = cmloc                                           &
     &                     + (xmid - xr)*dcy - (ymid - yr)*dcx

#else
                clloc = clloc - dcx*sna + dcy*csa
                cdloc = cdloc + dcx*csa + dcy*sna
                cmloc = cmloc                                           &
     &                     + (xmid - xr)*dcy - (ymid - yr)*dcx
#endif
               endif

 30    continue
!
! Viscous boundary
!
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
!
!  norm points outward, away from grid interior.
!  norm magnitude is area of surface triangle.
!
               xnorm =-0.5d0*(ay*bz - az*by)
               ynorm = 0.5d0*(ax*bz - az*bx)
               znorm =-0.5d0*(ax*by - ay*bx)

               p1    = qnod(1,node1)
               u1    = qnod(2,node1)
               v1    = qnod(3,node1)
               w1    = qnod(4,node1)

               p2    = qnod(1,node2)
               u2    = qnod(2,node2)
               v2    = qnod(3,node2)
               w2    = qnod(4,node2)

               p3    = qnod(1,node3)
               u3    = qnod(2,node3)
               v3    = qnod(3,node3)
               w3    = qnod(4,node3)

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
                cmloc = cmloc                                            &
     &                     + (xmid - xr)*dcy - (ymid - yr)*dcx
               endif

 60   continue
!
      icount = 3
      coef_loc(1) = clloc
      coef_loc(2) = cdloc
      coef_loc(3) = cmloc
      call MPI_ALLREDUCE(coef_loc,coef_glo,icount,                       &
     &                   MPIU_SCALAR, MPIU_SUM,                           &
     &                   MPI_COMM_WORLD,ierr)
!     call MPI_ALLREDUCE(cdloc,cdglo,icount,
!    &                   MPIU_REAL_PRECISION, MPI_SUM,
!    &                   MPI_COMM_WORLD,ierr)
!     call MPI_ALLREDUCE(cmloc,cmglo,icount,
!    &                   MPIU_REAL_PRECISION, MPI_SUM,
!    &                   MPI_COMM_WORLD,ierr)
!
       clw(ntt) = coef_glo(1)
       cdw(ntt) = coef_glo(2)
       cmw(ntt) = coef_glo(3)
       clift = clw(ntt)
       cdrag = cdw(ntt)
       cmom = cmw(ntt)
!
! Update the timing information - Added by D. K. Kaushik (1/17/97)
      xres(ntt) = tot
!
! End of subroutine FORCE
!
      return
      end

!================================ DELTAT2 ============================72
!
! Calculate a time step for each cell
! Note that this routine assumes conservative variables
!
!=====================================================================72
      subroutine DELTAT2(nnodes,nedge,qvec,cdt,                          &
     &                  xyz,vol,xyzn,evec,                               &
     &                  sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,             &
     &                  nsnode,nvnode,nfnode,isnode,ivnode,ifnode,       &
     &                  LocalTS,irank,nvertices)
!
      implicit none
#include <petsc/finclude/petscsys.h>
      PetscScalar  title(20),beta,alpha
      PetscScalar  Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      PetscScalar  gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,                  &
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      integer nnodes,nedge,nsnode,nvnode,nfnode,LocalTS,irank,nvertices
      PetscScalar  vol(1)
      PetscScalar  sxn(1),syn(1),szn(1)
      PetscScalar  vxn(1),vyn(1),vzn(1)
      PetscScalar  fxn(1),fyn(1),fzn(1)
      PetscScalar  cdt(1)
      integer isnode(1),ivnode(1),ifnode(1)
      integer n,i,node1,node2,inode
      integer ierr
      PetscLogDouble flops
      PetscScalar xnorm,ynorm,znorm,area
      PetscScalar u1,v1,w1,u2,v2,w2,u,v,c,w
      PetscScalar ubar,Vn,term
#if defined(INTERLACING)
      PetscScalar qvec(4,nvertices)
      PetscScalar xyz(3,nvertices)
      PetscScalar xyzn(4,nedge)
      integer evec(2,nedge)
#define qnod(i,j) qvec(i,j)
#define x(i) xyz(1,i)
#define y(i) xyz(2,i)
#define z(i) xyz(3,i)
#define xn(i) xyzn(1,i)
#define yn(i) xyzn(2,i)
#define zn(i) xyzn(3,i)
#define rl(i) xyzn(4,i)
#define eptr(j,i) evec(i,j)
#else
      PetscScalar qvec(nvertices,4)
      PetscScalar xyz(nvertices,3)
      PetscScalar xyzn(nedge,4)
      integer evec(nedge,2)
#define qnod(i,j) qvec(j,i)
#define x(i) xyz(i,1)
#define y(i) xyz(i,2)
#define z(i) xyz(i,3)
#define xn(i) xyzn(i,1)
#define yn(i) xyzn(i,2)
#define zn(i) xyzn(i,3)
#define rl(i) xyzn(i,4)
#define eptr(i,j) evec(i,j)
#endif
!
!  If local time steping, loop over faces
!  and calculate time step as cdt = V/(sum(|u.n| +c.area)
!  This is time step for cfl=1. We will multiply by cfl number later
!
! First loop over nodes and zero out cdt
!
      flops = 0.0
      if (LocalTS .gt. 0) then
         do 1000 i = 1,nvertices
            cdt(i) = 0.0d0
 1000    continue
!
         do 1020 n = 1, nedge
               node1 = eptr(n,1)
               node2 = eptr(n,2)
!
! Get normal to face
!
               xnorm = xn(n)
               ynorm = yn(n)
               znorm = zn(n)
               area  = rl(n)
!
               xnorm = xnorm*area
               ynorm = ynorm*area
               znorm = znorm*area
!
!/*
! xnorm = xnormal * area of face
! ynorm = ynormal * area of face
! znorm = znormal * area of face
!*/
               u1   = qnod(2,node1)
               v1   = qnod(3,node1)
               w1   = qnod(4,node1)

               u2   = qnod(2,node2)
               v2   = qnod(3,node2)
               w2   = qnod(4,node2)
!
! Get average values on face
!
               u    = 0.5d0*(u1 + u2)
               v    = 0.5d0*(v1 + v2)
               w    = 0.5d0*(w1 + w2)
               ubar = xn(n)*u + yn(n)*v + zn(n)*w
               c    = sqrt(ubar*ubar + beta)

               term = abs(u*xnorm + v*ynorm + w*znorm) + c*area
               cdt(node1) = cdt(node1) + term
               cdt(node2) = cdt(node2) + term
!
 1020    continue
         flops = flops + 27.0*nedge
!
! Now loop over boundaries and close the contours
!
         do 1030 i = 1,nsnode
            inode = isnode(i)
!
! Get the normal
!
            xnorm = sxn(i)
            ynorm = syn(i)
            znorm = szn(i)
            area  = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)

            u   = qnod(2,inode)
            v   = qnod(3,inode)
            w   = qnod(4,inode)
            ubar= (u*xnorm + v*ynorm + w*znorm)/area
            c   = sqrt(ubar*ubar + beta)
!
            Vn = abs(xnorm*u + ynorm*v + znorm*w) + c*area
            cdt(inode) = cdt(inode) + Vn
!
 1030    continue
         flops = flops + 24.0*nsnode

!
! Now viscous faces
!
!
         do 1040 i = 1,nvnode
            inode = ivnode(i)
!
! Get the normal
!
            xnorm = vxn(i)
            ynorm = vyn(i)
            znorm = vzn(i)
            area  = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
!
! Because velocities are zero, c is just sqrt(beta)
!
            c  = sqrt(beta)
            Vn = c*area

            cdt(inode) = cdt(inode) + Vn
!
 1040    continue
         flops = flops + 9.0*nvnode
!
! Now far field
!
         do 1050 i = 1,nfnode
            inode = ifnode(i)
!
! Get the normal
!
            xnorm = fxn(i)
            ynorm = fyn(i)
            znorm = fzn(i)
            area  = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)

            u   = qnod(2,inode)
            v   = qnod(3,inode)
            w   = qnod(4,inode)
            ubar= (u*xnorm + v*ynorm + w*znorm)/area
            c = sqrt(ubar*ubar + beta)

            Vn = abs(xnorm*u + ynorm*v + znorm*w) + c*area
            cdt(inode) = cdt(inode) + Vn
 1050    continue
         flops = flops + 24.0*nfnode

         do 1060 n = 1,nvertices
            cdt(n) = vol(n)/cdt(n)
 1060    continue
         flops = flops + nvertices
      else
!
! If not doing local time stepping just set cdt=1
!
         do 1070 n = 1,nvertices
            cdt(n) = 1.0d0
 1070    continue
      end if
      call PetscLogFlops(flops,ierr)
!
! End of subroutine DELTAT2
!
      return
      end

!================================= FLUX ======================================
!
! Calculates the fluxes on the face and performs the flux balance
!
!=============================================================================
#if defined(_OPENMP)
#if defined(HAVE_EDGE_COLORING)
      subroutine FLUX(nnodes, ncell, nedge,                                     &
     &                nsface, nvface, nfface, isface, ivface, ifface,           &
     &                nsnode, nvnode, nfnode, isnode, ivnode, ifnode,           &
     &                nnfacet,f2ntn,nnbound,                                    &
     &                nvfacet,f2ntv,nvbound,                                    &
     &                nffacet,f2ntf,nfbound,                                    &
     &                grad,evec,qvec,xyz,resvec,xyzn,                           &
     &                sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,phi,                  &
     &                max_threads,                                              &
     &                ncolor,ncount,                                            &
     &                irank,nvertices)
#elif defined(HAVE_REDUNDANT_WORK)
      subroutine FLUX(nnodes, ncell, nedge,                                     &
     &                nsface, nvface, nfface, isface, ivface, ifface,           &
     &                nsnode, nvnode, nfnode, isnode, ivnode, ifnode,           &
     &                nnfacet,f2ntn,nnbound,                                    &
     &                nvfacet,f2ntv,nvbound,                                    &
     &                nffacet,f2ntf,nfbound,                                    &
     &                grad,evec,qvec,xyz,resvec,xyzn,                           &
     &                sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,phi,                  &
     &                max_threads,                                              &
     &                resd,                                                     &
     &                irank,nvertices)
#else
      subroutine FLUX(nnodes, ncell, nedge,                                     &
     &                nsface, nvface, nfface, isface, ivface, ifface,           &
     &                nsnode, nvnode, nfnode, isnode, ivnode, ifnode,           &
     &                nnfacet,f2ntn,nnbound,                                    &
     &                nvfacet,f2ntv,nvbound,                                    &
     &                nffacet,f2ntf,nfbound,                                    &
     &                grad,evec,qvec,xyz,resvec,xyzn,                           &
     &                sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,phi,                  &
     &                max_threads,                                              &
     &                nedgeAllThr,part_thr,nedge_thr,edge_thr,xyzn_thr,         &
     &                irank,nvertices)
#endif
#else
      subroutine FLUX(nnodes, ncell, nedge,                                     &
     &                nsface, nvface, nfface, isface, ivface, ifface,           &
     &                nsnode, nvnode, nfnode, isnode, ivnode, ifnode,           &
     &                nnfacet,f2ntn,nnbound,                                    &
     &                nvfacet,f2ntv,nvbound,                                    &
     &                nffacet,f2ntf,nfbound,                                    &
     &                grad,evec,qvec,xyz,resvec,xyzn,                           &
     &                sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,phi,                  &
     &                irank,nvertices)
#endif
      implicit none
#include <petsc/finclude/petscsys.h>
      PetscScalar  title(20),beta,alpha
      PetscScalar  Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      PetscScalar  gamma,gm1,gp1,gm1g,gp1g,ggm1
      PetscScalar  p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      PetscScalar  rms(1001),clw(1001),cdw(1001)
      PetscScalar  cmw(1001),xres(1001)
      PetscScalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,                                &
     &        nstage,ncyct,iramp,nitfo
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,                         &
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/history/rms,clw,cdw,cmw,xres
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,                 &
     &             nstage,ncyct,iramp,nitfo
      integer nnodes, ncell, nedge,                                             &
     &        nsface, nvface, nfface,                                           &
     &        nsnode, nvnode, nfnode,                                           &
     &        nnfacet,nvfacet,nffacet,                                          &
     &        nnbound,nvbound,nfbound,                                          &
     &        irank,nvertices
      PetscScalar  sxn(1),syn(1),szn(1)
      PetscScalar  vxn(1),vyn(1),vzn(1)
      PetscScalar  fxn(1),fyn(1),fzn(1)
      PetscScalar  phi(nvertices,4)
      integer isface(1),ivface(1),ifface(1)
      integer isnode(1),ivnode(1),ifnode(1)
      integer f2ntn(nnfacet,4)
      integer f2ntv(nvfacet,4)
      integer f2ntf(nffacet,4)
      integer i,n,node1,node2,node3,inode
      integer ierr,first_edge,last_edge
      PetscLogDouble flops
      PetscScalar  xmean,ymean,zmean,xnorm,ynorm
      PetscScalar  znorm,rlen,area,dot,size
      PetscScalar  X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3
      PetscScalar  p1,p2,p3
      PetscScalar  rx,ry,rz,pL,uL,wL,vL
      PetscScalar  pR,uR,vR,wR
      PetscScalar  ubarL,ubarR,ubar,p,u,v,w,c
      PetscScalar  c2,c20
      PetscScalar  ubar0,pi,ui,vi,wi,unorm
      PetscScalar  phi1,phi2,phi3,phi4,phi5
      PetscScalar  phi6,phi7,phi8,phi9
      PetscScalar  eig1,eig2,eig3,eig4
      PetscScalar  dp,du,dv,dw
      PetscScalar  ti11,ti21,ti31,ti41
      PetscScalar  dv1,dv2,dv3,dv4
      PetscScalar  r11,r21,r31,r41
      PetscScalar  r12,r22,r32,r42
      PetscScalar  r13,r23,r33,r43
      PetscScalar  r14,r24,r34,r44
      PetscScalar  t1,t2,t3,t4
      PetscScalar  t11,t21,t31,t41
      PetscScalar  t12,t22,t32,t42
      PetscScalar  t13,t23,t33,t43
      PetscScalar  t14,t24,t34,t44
      PetscScalar  fluxp1,fluxp2,fluxp3,fluxp4
      PetscScalar  fluxm1,fluxm2,fluxm3,fluxm4
      PetscScalar  res1,res2,res3,res4
      PetscScalar  rhs1,rhs2,rhs3,rhs4
      PetscScalar  c68,c18,second
      PetscScalar  ax,ay,az,bx,by,bz
      PetscScalar  pa,pb,pc,ub,vb,wb
#if defined(INTERLACING)
      PetscScalar qvec(4,nvertices)
      PetscScalar grad(3,4,nvertices)
      PetscScalar resvec(4,nnodes)
      PetscScalar xyz(3,nvertices)
      PetscScalar xyzn(4,nedge)
      integer evec(2,nedge)
#define qnod(i,j) qvec(i,j)
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
      PetscScalar qvec(nvertices,4)
      PetscScalar grad(nvertices,4,3)
      PetscScalar resvec(nnodes,4)
      PetscScalar xyz(nvertices,3)
      PetscScalar xyzn(nedge,4)
      integer evec(nedge,2)
#define qnod(i,j) qvec(j,i)
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
       integer my_thread_id,omp_get_thread_num,                             &
     &         max_threads,omp_get_num_threads,                             &
     &         first_node,last_node,chunk
#if defined(HAVE_EDGE_COLORING)
       integer ncolor,ncount(1)
#elif defined(HAVE_REDUNDANT_WORK)
       PetscScalar resd(4,nnodes)
#undef res
#define res(a,b) resd(a,b)
#else
       integer nedgeAllThr,part_thr(nnodes),nedge_thr(0:max_threads),         &
     &     edge_thr(2,nedgeAllThr)
       PetscScalar xyzn_thr(4,nedgeAllThr)
#undef eptr
#define eptr(a,b) edge_thr(b,a)
#undef xn
#undef yn
#undef zn
#undef ra
#define xn(i) xyzn_thr(1,i)
#define yn(i) xyzn_thr(2,i)
#define zn(i) xyzn_thr(3,i)
#define ra(i) xyzn_thr(4,i)
#endif
#endif
!      logging variables
       integer flux_event, fun3d_class, flag
       character * 16 flux_label
       data flag/-1/,flux_label/'FluxEvaluation  '/
       save flux_event, fun3d_class, flag, flux_label
       if (flag .eq. -1) then
          call PetscClassIdRegister('PetscFun3d',fun3d_class,ierr)
          call PetscLogEventRegister(flux_label,                        &
     &         fun3d_class,flux_event,ierr)
          flag = 1
       endif
       call PetscLogEventBegin(flux_event,ierr)
       flops = 0.0
       second = 1.0d0
!      if (ntt.le.nitfo)second = 0.0
!      print *, 'Second is' , second
#if defined(_OPENMP)
#if defined(HAVE_EDGE_COLORING)
       first_edge = 1
       do i = 1,ncolor
             last_edge = first_edge + ncount(i) - 1
#elif defined(HAVE_REDUNDANT_WORK)
       do i = 1,nnodes
          do j = 1,4
             resd(j,i) = 0.0
          enddo
       enddo
       first_edge = 1
       last_edge = nedge
#endif
#else
       first_edge = 1
       last_edge = nedge
#endif
!
! Loop over all the faces and calculate flux
!
!$omp  parallel default(shared) private(n,node1,node2,xmean,ymean,           &
!$omp& zmean,xnorm,ynorm,znorm,rlen,dot,x1,y1,z1,size,x2,y2,z2,rx,           &
!$omp& ry,rz,pL,uL,vL,wL,ubarL,pR,uR,vR,wR,ubarR,p,u,v,w,ubar,phi1,          &
!$omp& phi2,phi3,phi4,phi5,phi6,phi7,phi8,phi9,c2,c,eig1,eig2,eig3,          &
!$omp& eig4,dp,du,dv,dw,ti11,ti21,ti31,ti41,dv1,dv2,dv3,dv4,r11,r21,         &
!$omp& r31,r41,r12,r22,r32,r42,r13,r23,r33,r43,r14,r24,r34,r44,t1,t2,        &
!$omp& t3,t4,fluxp1,fluxp2,fluxp3,fluxp4,fluxm1,fluxm2,fluxm3,fluxm4,        &
!$omp& res1,res2,res3,res4,c68,c18,inode)                                    &
!$omp& private(x3,y3,z3,p1,p2,p3,ax,ay,az,bx,by,bz,pa,pb,pc,area,c20,        &
!$omp& c0,ubar0,t11,t21,t31,t41,t12,t22,t32,t42,t13,t23,t33,t43,t14,         &
!$omp& t24,t34,t44,pi,ui,vi,wi,unorm,rhs1,rhs2,rhs3,rhs4,ub,vb,wb,node3)     &
!$omp& private(my_thread_id,chunk)                                           &
!$omp& reduction(+:flops)                                                    &
#if defined(_OPENMP)
#if defined(HAVE_EDGE_COLORING)
#elif defined(HAVE_REDUNDANT_WORK)
!$omp& firstprivate(resd)
#else
!$omp& private(first_edge,last_edge)
#endif
#endif
!!$omp1 shared(evec,qvec,resvec,xyz,xyzn,grad,nedge,nnodes,second,flops,
!!$omp2 title,beta,alpha,Re,dt,tot,res0,resc,ntt,mseq,ivisc,irest,icyc,
!!$omp3 ihane,ntturb,gamma,gm1,gp1,gm1g,gp1g,ggm1,p0,rho0,c0,u0,v0,w0,
!!$omp4 et0,h0,pt0,rms,clw,cdw,cmw,xres,cfl1,cfl2,nsmoth,iflim,
!!$omp5 itran,nbtran,jupdate,nstage,ncyct,iramp,nitfo)
!!$      my_thread_id = omp_get_thread_num()
!!$omp   critical
!!$      print *, 'My thread id on processor ',irank,' is ',my_thread_id
!!$omp   end critical
#if defined(_OPENMP)
#if defined(HAVE_EDGE_COLORING)
!$omp do
#elif defined(HAVE_REDUNDANT_WORK)
!$omp do
#else
!     tot_threads = omp_get_num_threads()
      my_thread_id = omp_get_thread_num()
!     chunk = nnodes/tot_threads
!     first_node = my_thread_id*chunk+1
!     if (my_thread_id .eq. (size-1)) then
!        last_node = nnodes
!     else
!        last_node = (my_thread_id+1)*chunk
!     endif
      first_edge = nedge_thr(my_thread_id)
      last_edge = nedge_thr(my_thread_id+1)-1

!      print *, 'On thread ', my_thread_id,', first_edge = ',first_edge,        &
!     &         ', last_edge = ',last_edge

#endif
#else
#endif
      do n = first_edge, last_edge
        node1 = eptr(n,1)
        node2 = eptr(n,2)
#if defined(_OPENMP) && !defined(HAVE_REDUNDANT_WORK) && !defined(HAVE_EDGE_COLORING)
        if ((part_thr(node1) .eq. my_thread_id)                                 &
     &       .or.(part_thr(node2) .eq. my_thread_id)) then
#endif
!
! Calculate unit normal to face and length of face
!
          xmean = .5d0*(x(node1) + x(node2))
          ymean = .5d0*(y(node1) + y(node2))
          zmean = .5d0*(z(node1) + z(node2))
          xnorm  = xn(n)
          ynorm  = yn(n)
          znorm  = zn(n)
          rlen   = ra(n)
!
! Now lets get our other 2 vectors
! For first vector, use {1,0,0} and subtract off the component
! in the direction of the face normal. If the inner product of
! {1,0,0} is close to unity, use {0,1,0}
!
         dot = xnorm
         if (abs(dot).lt.0.95d0)then
          X1 = 1. - dot*xnorm
          Y1 =    - dot*ynorm
          Z1 =    - dot*znorm
         else
          dot = ynorm
          X1 =    - dot*xnorm
          Y1 = 1.0d0 - dot*ynorm
          Z1 =    - dot*znorm
         end if
!
! Normalize the first vector
!
         size = sqrt(X1*X1 + Y1*Y1 + Z1*Z1)
         X1 = X1/size
         Y1 = Y1/size
         Z1 = Z1/size
!
! Take cross-product of normal and V1 to get V2
!
         X2 = ynorm*Z1 - znorm*Y1
         Y2 = znorm*X1 - xnorm*Z1
         Z2 = xnorm*Y1 - ynorm*X1
!
! Get variables on "left" and "right" side of face
!
         rx     = second*(xmean - x(node1))
         ry     = second*(ymean - y(node1))
         rz     = second*(zmean - z(node1))
         pL = qnod(1,node1) + gradx(1,node1)*rx                           &
     &                       + grady(1,node1)*ry                           &
     &                       + gradz(1,node1)*rz
         uL = qnod(2,node1) + gradx(2,node1)*rx                           &
     &                       + grady(2,node1)*ry                           &
     &                       + gradz(2,node1)*rz
         vL = qnod(3,node1) + gradx(3,node1)*rx                           &
     &                       + grady(3,node1)*ry                           &
     &                       + gradz(3,node1)*rz
         wL = qnod(4,node1) + gradx(4,node1)*rx                           &
     &                       + grady(4,node1)*ry                           &
     &                       + gradz(4,node1)*rz

         ubarL  = xnorm*uL + ynorm*vL + znorm*wL
!        c2L = ubarl*ubarL + beta
!        cL  = sqrt(c2L)
!
         rx     = second*(xmean - x(node2))
         ry     = second*(ymean - y(node2))
         rz     = second*(zmean - z(node2))
         pR   = qnod(1,node2) + gradx(1,node2)*rx                         &
     &                         + grady(1,node2)*ry                         &
     &                         + gradz(1,node2)*rz
         uR   = qnod(2,node2) + gradx(2,node2)*rx                         &
     &                         + grady(2,node2)*ry                         &
     &                         + gradz(2,node2)*rz
         vR   = qnod(3,node2) + gradx(3,node2)*rx                         &
     &                         + grady(3,node2)*ry                         &
     &                         + gradz(3,node2)*rz
         wR   = qnod(4,node2) + gradx(4,node2)*rx                         &
     &                         + grady(4,node2)*ry                         &
     &                         + gradz(4,node2)*rz
         ubarR  = xnorm*uR + ynorm*vR + znorm*wR
!        c2R = ubarR*ubarR + beta
!        cR  = sqrt(c2R)
!
! Compute averages
!
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
!
! Now compute eigenvalues, eigenvectors, and strengths
!
          eig1 = abs(ubar)
          eig2 = abs(ubar)
          eig3 = abs(ubar + c)
          eig4 = abs(ubar - c)
!
          dp = pr - pl
          du = ur - ul
          dv = vr - vl
          dw = wr - wl
!
! Components of T(inverse) (I will divide by c2 later)
!
          ti11 = -(u*phi4 + v*phi5 + w*phi6)/beta
          ti21 = -(u*phi7 + v*phi8 + w*phi9)/beta
          ti31 =  .5d0*(c-ubar)/beta
          ti41 = -.5d0*(c+ubar)/beta
!
! jumps (T(inverse)*dq)
          dv1 = (ti11*dp + phi4*du + phi5*dv + phi6*dw)/c2
          dv2 = (ti21*dp + phi7*du + phi8*dv + phi9*dw)/c2
          dv3 = .5d0*(2.d0*ti31*dp + xnorm*du + ynorm*dv + znorm*dw)/c2
          dv4 = .5d0*(2.d0*ti41*dp + xnorm*du + ynorm*dv + znorm*dw)/c2
!
! Now get elements of T (call it r for now)
!
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
!
          r14 = -c*beta
          r24 = xnorm*beta + u*(ubar - c)
          r34 = ynorm*beta + v*(ubar - c)
          r44 = znorm*beta + w*(ubar - c)
!
! Calculate T* |lambda| *T(inverse)
!
          t1 = eig1*r11*dv1 + eig2*r12*dv2 + eig3*r13*dv3 + eig4*r14*dv4
          t2 = eig1*r21*dv1 + eig2*r22*dv2 + eig3*r23*dv3 + eig4*r24*dv4
          t3 = eig1*r31*dv1 + eig2*r32*dv2 + eig3*r33*dv3 + eig4*r34*dv4
          t4 = eig1*r41*dv1 + eig2*r42*dv2 + eig3*r43*dv3 + eig4*r44*dv4
!
! Modify to calculate .5(fl +fr) from nodes
! instead of extrapolated ones
!
!         pL    = qnod(1,node1)
!         uL    = qnod(2,node1)
!         vL    = qnod(3,node1)
!         wL    = qnod(4,node1)
!         ubarL = xnorm*uL + ynorm*vL + znorm*wL
!
          fluxp1 = rlen*beta*ubarL
          fluxp2 = rlen*(uL*ubarL + xnorm*pL)
          fluxp3 = rlen*(vL*ubarL + ynorm*pL)
          fluxp4 = rlen*(wL*ubarL + znorm*pL)
!
! Now the right side
!
!         pR    = qnod(1,node2)
!         uR    = qnod(2,node2)
!         vR    = qnod(3,node2)
!         wR    = qnod(4,node2)
!         ubarR = xnorm*uR + ynorm*vR + znorm*wR
!
          fluxm1 = rlen*beta*ubarR
          fluxm2 = rlen*(uR*ubarR + xnorm*pR)
          fluxm3 = rlen*(vR*ubarR + ynorm*pR)
          fluxm4 = rlen*(wR*ubarR + znorm*pR)
!
          res1 = 0.5d0*(fluxp1 + fluxm1 - rlen*t1)
          res2 = 0.5d0*(fluxp2 + fluxm2 - rlen*t2)
          res3 = 0.5d0*(fluxp3 + fluxm3 - rlen*t3)
          res4 = 0.5d0*(fluxp4 + fluxm4 - rlen*t4)
!
          flops = flops + 318.0

!
#if defined(_OPENMP) && !defined(HAVE_REDUNDANT_WORK) && !defined(HAVE_EDGE_COLORING)
          if ((part_thr(node1).eq.my_thread_id)                                 &
     &        .and.(node1.le.nnodes)) then
#else
          if (node1.le.nnodes) then
#endif
           res(1,node1) = res(1,node1) + res1
           res(2,node1) = res(2,node1) + res2
           res(3,node1) = res(3,node1) + res3
           res(4,node1) = res(4,node1) + res4
           flops = flops + 4.0
          endif
!
#if defined(_OPENMP) && !defined(HAVE_REDUNDANT_WORK) && !defined(HAVE_EDGE_COLORING)
          if ((part_thr(node2).eq.my_thread_id)                                 &
     &        .and.(node2.le.nnodes)) then
#else
          if (node2.le.nnodes) then
#endif
           res(1,node2) = res(1,node2) - res1
           res(2,node2) = res(2,node2) - res2
           res(3,node2) = res(3,node2) - res3
           res(4,node2) = res(4,node2) - res4
           flops = flops + 4.0
          endif
#if defined(_OPENMP) && !defined(HAVE_REDUNDANT_WORK)&& !defined(HAVE_EDGE_COLORING)
         endif
#endif
!
      enddo
#if defined(_OPENMP)
#if defined(HAVE_EDGE_COLORING)
!$omp enddo
!$omp end parallel
       first_edge = first_edge + ncount(i)
      enddo
#elif defined(HAVE_REDUNDANT_WORK)
!$omp enddo
!$omp critical (residual_update)
       do n = 1,nnodes
          resvec(1,n)=resvec(1,n)+res(1,n)
          resvec(2,n)=resvec(2,n)+res(2,n)
          resvec(3,n)=resvec(3,n)+res(3,n)
          resvec(4,n)=resvec(4,n)+res(4,n)
       enddo
       flops = flops + 4.0*nnodes
!$omp end critical (residual_update)
!$omp end parallel
#undef res
#define res(a,b) resvec(a,b)
#else
!$omp end parallel
#endif
#else
#endif
      call PetscLogFlops(flops,ierr)
      call PetscLogEventEnd(flux_event,ierr)
!     c68 = 6./8.
!     c18 = 1./8.
      c68 = 0.75d0
      c18 = 0.125d0
!
! Close contour over the boundaries
! First do inviscid faces
!
      do n = 1, nnfacet
               node1 = isnode(f2ntn(n,1))
               node2 = isnode(f2ntn(n,2))
               node3 = isnode(f2ntn(n,3))

               x1 = x(node1)
               y1 = y(node1)
               z1 = z(node1)
               p1 = qnod(1,node1)

               x2 = x(node2)
               y2 = y(node2)
               z2 = z(node2)
               p2 = qnod(1,node2)

               x3 = x(node3)
               y3 = y(node3)
               z3 = z(node3)
               p3 = qnod(1,node3)

               ax = x2 - x1
               ay = y2 - y1
               az = z2 - z1

               bx = x3 - x1
               by = y3 - y1
               bz = z3 - z1
!
! Normal points away from grid interior.
! Magnitude is 1/3 area of surface triangle.
!
               xnorm =-0.5d0*(ay*bz - az*by)/3.d0
               ynorm = 0.5d0*(ax*bz - az*bx)/3.d0
               znorm =-0.5d0*(ax*by - ay*bx)/3.d0

               pa = c68*p1 + c18*(p2 + p3)
               pb = c68*p2 + c18*(p3 + p1)
               pc = c68*p3 + c18*(p1 + p2)
!
               flops = flops + 35.0
          if (node1.le.nnodes) then
               res(2,node1) = res(2,node1) + xnorm*pa
               res(3,node1) = res(3,node1) + ynorm*pa
               res(4,node1) = res(4,node1) + znorm*pa
               flops = flops + 6.0
          endif

          if (node2.le.nnodes) then
               res(2,node2) = res(2,node2) + xnorm*pb
               res(3,node2) = res(3,node2) + ynorm*pb
               res(4,node2) = res(4,node2) + znorm*pb
               flops = flops + 6.0
          endif

          if (node3.le.nnodes) then
               res(2,node3) = res(2,node3) + xnorm*pc
               res(3,node3) = res(3,node3) + ynorm*pc
               res(4,node3) = res(4,node3) + znorm*pc
               flops = flops + 6.0
          endif

      enddo
!
! Now viscous faces
!
      do n = 1, nvfacet
               node1 = ivnode(f2ntv(n,1))
               node2 = ivnode(f2ntv(n,2))
               node3 = ivnode(f2ntv(n,3))

               x1 = x(node1)
               y1 = y(node1)
               z1 = z(node1)
               p1 = qnod(1,node1)

               x2 = x(node2)
               y2 = y(node2)
               z2 = z(node2)
               p2 = qnod(1,node2)

               x3 = x(node3)
               y3 = y(node3)
               z3 = z(node3)
               p3 = qnod(1,node3)

               ax = x2 - x1
               ay = y2 - y1
               az = z2 - z1

               bx = x3 - x1
               by = y3 - y1
               bz = z3 - z1
!
! norm point away from grid interior.
! norm magnitude is 1/3 area of surface triangle.
!
               xnorm =-0.5d0*(ay*bz - az*by)/3.d0
               ynorm = 0.5d0*(ax*bz - az*bx)/3.d0
               znorm =-0.5d0*(ax*by - ay*bx)/3.d0

               pa = c68*p1 + c18*(p2 + p3)
               pb = c68*p2 + c18*(p3 + p1)
               pc = c68*p3 + c18*(p1 + p2)
!
               flops = flops + 35.0
!
          if (node1.le.nnodes) then
                res(2,node1) = res(2,node1) + xnorm*pa
                res(3,node1) = res(3,node1) + ynorm*pa
                res(4,node1) = res(4,node1) + znorm*pa
                flops = flops + 6.0

          endif

          if (node2.le.nnodes) then
                res(2,node2) = res(2,node2) + xnorm*pb
                res(3,node2) = res(3,node2) + ynorm*pb
                res(4,node2) = res(4,node2) + znorm*pb
                flops = flops + 6.0
          endif

          if (node3.le.nnodes) then
                res(2,node3) = res(2,node3) + xnorm*pc
                res(3,node3) = res(3,node3) + ynorm*pc
                res(4,node3) = res(4,node3) + znorm*pc
                flops = flops + 6.0
          endif

      enddo
!
! The next section of code is for when you dont care about
! preserving linear data on boundary. Also, doing this
! matches the left hand side when not doing Newton-Krylov
! Usually just go around unless you are just experimenting
!
      goto 1025
!
! Loop over the boundaries
! First do inviscid nodes
!
!     do 1020 i = 1,nsnode
!       inode = isnode(i)
!
!       xnorm = sxn(i)
!       ynorm = syn(i)
!       znorm = szn(i)
!
!       p = qnod(1,inode)
!
!       if (inode .le. nnodes) then
!        res(2,inode) = res(2,inode) + xnorm*p
!        res(3,inode) = res(3,inode) + ynorm*p
!        res(4,inode) = res(4,inode) + znorm*p
!       endif
!
!1020 continue
!
! Now viscous nodes
!
!     do 1030 i = 1,nvnode
!       inode = ivnode(i)
!
!       xnorm = vxn(i)
!       ynorm = vyn(i)
!       znorm = vzn(i)
!
!       p = qnod(1,inode)
!
!       if (inode .le. nnodes) then
!        res(2,inode) = res(2,inode) + xnorm*p
!        res(3,inode) = res(3,inode) + ynorm*p
!        res(4,inode) = res(4,inode) + znorm*p
!       endif
!
!1030 continue
!
 1025 continue
!
! Now do far-field
!
       do i = 1,nfnode
         inode   = ifnode(i)
!
!/*
! Get normal and "other" 2 vectors. Remember that fxn,fyn and fzn
! has the magnitude of the face contained in it.
!*/
         xnorm   = fxn(i)
         ynorm   = fyn(i)
         znorm   = fzn(i)
         area    = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
         xnorm   = xnorm/area
         ynorm   = ynorm/area
         znorm   = znorm/area
!
! Now lets get our other 2 vectors
! For first vector, use {1,0,0} and subtract off the component
! in the direction of the face normal. If the inner product of
! {1,0,0} is close to unity, use {0,1,0}
!
         dot = xnorm
         if (abs(dot).lt.0.95d0)then
          X1 = 1.d0 - dot*xnorm
          Y1 =    - dot*ynorm
          Z1 =    - dot*znorm
         else
          dot = ynorm
          X1 =    - dot*xnorm
          Y1 = 1.d0 - dot*ynorm
          Z1 =    - dot*znorm
         end if
!
! Normalize the first vector (V1)
!
         size = sqrt(X1*X1 + Y1*Y1 + Z1*Z1)
         X1 = X1/size
         Y1 = Y1/size
         Z1 = Z1/size
!
! Take cross-product of normal with V1 to get V2
!
         X2 = ynorm*Z1 - znorm*Y1
         Y2 = znorm*X1 - xnorm*Z1
         Z2 = xnorm*Y1 - ynorm*X1

!
! Calculate elements of T and T(inverse) evaluated at freestream
!
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
!
! Now, get the variables on the "inside"
!
         pi      = qnod(1,inode)
         ui      = qnod(2,inode)
         vi      = qnod(3,inode)
         wi      = qnod(4,inode)
         unorm   = xnorm*ui + ynorm*vi + znorm*wi
!
! If ubar is negative, take the reference condition from outside
!
!
         if (unorm.gt.0.0d0)then
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
!
! Set rhs
!
         rhs1 = (ti11*pr + phi4*ur + phi5*vr + phi6*wr)/c20
         rhs2 = (ti21*pr + phi7*ur + phi8*vr + phi9*wr)/c20
         rhs3 = .5d0*(2.d0*ti31*pi + xnorm*ui + ynorm*vi + znorm*wi)/c20
         rhs4 = .5d0*(2.d0*ti41*p0 + xnorm*u0 + ynorm*v0 + znorm*w0)/c20
!
! Now do matrix multiplication to get values on boundary
!
         pb =                       t13*rhs3 + t14*rhs4
         ub = t21*rhs1 + t22*rhs2 + t23*rhs3 + t24*rhs4
         vb = t31*rhs1 + t32*rhs2 + t33*rhs3 + t34*rhs4
         wb = t41*rhs1 + t42*rhs2 + t43*rhs3 + t44*rhs4

         ubar = xnorm*ub + ynorm*vb + znorm*wb
         flops = flops + 180.0
!
        if (inode.le.nnodes) then
          res(1,inode) = res(1,inode)+area*beta*ubar
          res(2,inode) = res(2,inode)+area*(ub*ubar + xnorm*pb)
          res(3,inode) = res(3,inode)+area*(vb*ubar + ynorm*pb)
          res(4,inode) = res(4,inode)+area*(wb*ubar + znorm*pb)
          flops = flops + 18.0
        endif
!
       enddo
       call PetscLogFlops(flops,ierr)
!
! End of subroutine FLUX
!
      return
      end

!---------------------------------------------------------------
! The following subroutines are from node3t.f in the original
! code - D. K. Kaushik (1/17/97)
!---------------------------------------------------------------
!
!=============================== SUMGS ===============================72
!
! Gets the weights for calculating gradients using least squares
!
!=====================================================================72
      subroutine SUMGS(nnodes,nedge,evec,xyz,rxy,irank,nvertices)

      implicit none
#include <petsc/finclude/petscsys.h>
      integer nnodes,nedge,irank,nvertices
#if defined(INTERLACING)
      PetscScalar xyz(3,nvertices)
      PetscScalar rxy(7,nnodes)
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
#undef eptr
#define eptr(j,i) evec(i,j)
#else
      PetscScalar xyz(nvertices,3)
      PetscScalar rxy(nnodes,7)
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
      PetscScalar x1,y1,z1,x2,y2,z2,dx,dy,dz
      PetscScalar dist,weight,w2,w11,w22,w33
      PetscScalar r12r11,r13r11,r23r22,rmess

!
! Initialize all the rij to 0.0
!
      do 1000 i = 1,nnodes
         r11(i) = 0.0
         r12(i) = 0.0
         r13(i) = 0.0
         r22(i) = 0.0
         r23(i) = 0.0
         r33(i) = 0.0
         r44(i) = 0.0
 1000 continue
!
! Now loop over the edges and accumulate the r
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
!         weight = 1.0d0/dist
          weight = 1.0d0
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
!/*
! Now calculate ||x(:)|| = r11 by taking the square root
! Also divide r12 and r13 by ||x(:)||
!*/
      do 1030 i = 1,nnodes
        r11(i) = sqrt(r11(i))
        r12(i) = r12(i)/r11(i)
        r13(i) = r13(i)/r11(i)
 1030 continue
!
!/*
! Now calculate r22 and r23
!*/
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
!
          dx = x2 - x1
          dy = y2 - y1
          dz = z2 - z1
          dist = sqrt(dx*dx + dy*dy + dz*dz)
!         weight = 1.0/dist
          weight = 1.0d0
          dx = weight*dx
          dy = weight*dy
          dz = weight*dz
          w2 = weight*weight
!
          if (node1 .le. nnodes) then
           r22(node1) = r22(node1) +                                         &
     &                  (dy-dx*r12(node1)/r11(node1))**2
           r23(node1) = r23(node1) + dz*                                     &
     &                  (dy-dx*r12(node1)/r11(node1))
          endif
!
          if (node2 .le. nnodes) then
           r22(node2) = r22(node2) +                                         &
     &                  (-dy + dx*r12(node2)/r11(node2))**2
           r23(node2) = r23(node2) - dz*                                     &
     &                  (-dy+dx*r12(node2)/r11(node2))
          endif
 1050 continue
!
!/*
! Now finish getting r22 and r23
!*/
      do 1060 i = 1,nnodes
        r22(i) = sqrt(r22(i))
        r23(i) = r23(i)/r22(i)
 1060 continue
!
!/*
! Now all we have to do is get r33
!*/
      do 1080 n = 1, nedge
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
!         weight = 1.0/dist
          weight = 1.0d0
          dx = weight*dx
          dy = weight*dy
          dz = weight*dz
          w2 = weight*weight
!
          if (node1 .le. nnodes) then
           r33(node1) = r33(node1) +                                        &
     &                  (dz-dx*r13(node1)/r11(node1)-                       &
     &                  r23(node1)/r22(node1)*                              &
     &                  (dy - dx*r12(node1)/                                &
     &                   r11(node1)))**2
          endif
!
          if (node2 .le. nnodes) then
           r33(node2) = r33(node2) +                                        &
     &                  (-dz+dx*r13(node2)/r11(node2)-                      &
     &                  r23(node2)/r22(node2)*                              &
     &                  (-dy + dx*r12(node2)/                               &
     &                  r11(node2)))**2
          endif
!
 1080 continue
!
!/*
! Now just get the magnitude of r33
!*/
      do 1090 i = 1,nnodes
        r33(i) = sqrt(r33(i))
 1090 continue
!
!/*
! Added by Dinesh Kaushik (6/27/97)
! The following addition changes the meaning of r11 .. r33. r44
! is the new parameter introduced by me. The new definitions
! are taken from LSTGS (where these parameters
! are used finally).
! r11->w11
! r22->w22
! r33->w33
! r12->r12r11
! r13->r13r11
! r23->r23r22
! r44->rmess */

      do i = 1,nnodes
           w11 = 1.d0/(r11(i)*r11(i))
           w22 = 1.d0/(r22(i)*r22(i))
           w33 = 1.d0/(r33(i)*r33(i))
           r12r11 = r12(i)/r11(i)
           r13r11 = r13(i)/r11(i)
           r23r22 = r23(i)/r22(i)
           rmess=(r12(i)*r23(i)-                                             &
     &            r13(i)*r22(i))/                                            &
     &           (r11(i)*r22(i)*                                             &
     &            r33(i)*r33(i))

           r11(i) = w11
           r22(i) = w22
           r33(i) = w33
           r12(i) = r12r11
           r13(i) = r13r11
           r23(i) = r23r22
           r44(i) = rmess
      enddo
!
! Finished with SUMGS
!
      return
      end

!================================= LSTGS =============================72
!
! Calculates the Gradients at the nodes using weighted least squares
! This subroutine solves using Gram-Schmidt
!
!=====================================================================72
      subroutine LSTGS(nnodes,nedge,evec,                                &
     &                 qvec,grad,xyz,                                    &
     &                 rxy,irank,nvertices)
!
      implicit none
#include <petsc/finclude/petscsys.h>
      integer nnodes,nedge,irank,nvertices
!
#if defined(INTERLACING)
      PetscScalar qvec(4,nvertices)
      PetscScalar xyz(3,nvertices)
      PetscScalar rxy(7,nnodes)
      PetscScalar grad(3,4,nnodes)
      integer evec(2,nedge)
#define qnod(i,j) qvec(i,j)
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
      PetscScalar qvec(nvertices,4)
      PetscScalar xyz(nvertices,3)
      PetscScalar rxy(nnodes,7)
      PetscScalar grad(nnodes,4,3)
      integer evec(nedge,2)
#define qnod(i,j) qvec(j,i)
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
      PetscScalar  title(20),beta,alpha
      PetscScalar  Re,dt,tot,res0,resc
      integer i,ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      PetscScalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,                          &
     &        nstage,ncyct,iramp,nitfo
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,                   &
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,           &
     &             nstage,ncyct,iramp,nitfo
      integer n,node1,node2,ierr
      PetscLogDouble flops
      PetscScalar  dx1,dy1,dz1,dx2,dy2,dz2,                                    &
     &        dq1,dq2,dq3,dq4,                                            &
     &        weight,w11,r12r11,r13r11,w22,r23r22,w33,rmess,              &
     &        coef1,coef2,termx,termy,termz
!     logging variables
!     integer flag
!     integer grad_event,node1_event,node2_event
!     character * 16 grad_label, node1_label, node2_label
!     data flag/-1/,grad_label/'GRAD            '/
!     data node1_label/'NODE1           '/
!     data node2_label/'NODE2           '/
!     save grad_event, grad_label, flag
!     save node1_event,node2_event,node1_label, node2_label

!     if (flag .eq. -1) then
!        call PetscLogEventRegister(grad_event,grad_label,ierr)
!        call PetscLogEventRegister(node1_event,node1_label,ierr)
!        call PetscLogEventRegister(node2_event,node2_label,ierr)
!        flag = 1
!     endif
!     call PetscLogEventBegin(grad_event,0,0,0,0,ierr)

      flops = 0.0
! For checking out the code input a linear distribution
!
!     write(6,700)nnodes,nedge
! 700 format(1h ,'nnodes=',i5,' nedge=',i5)
!     do 1001 i = 1,nnodes
!     write(6,800)i,x(i),y(i),z(i)
! 800 format(1h ,'i x y z=',i5,3(f10.5,1x))
!       qnod(1,i) = 1.0*x(i) +  2.0*y(i) + 3.0*z(i)
!       qnod(2,i) = 3.0*x(i) +  4.0*y(i) + 6.0*z(i)
!       qnod(3,i) = 5.0*x(i) +  6.0*y(i) + 9.0*z(i)
!       qnod(4,i) = 7.0*x(i) +  8.0*y(i) + 12.0*z(i)
!       qnod(i,5) = 9.0*x(i) + 10.0*y(i) + 15.0*z(i)
!1001 continue
!
! Zero out the gradients
!
      do 1000 i = 1,nnodes
         gradx(1,i) = 0.0
         grady(1,i) = 0.0
         gradz(1,i) = 0.0

         gradx(2,i) = 0.0
         grady(2,i) = 0.0
         gradz(2,i) = 0.0

         gradx(3,i) = 0.0
         grady(3,i) = 0.0
         gradz(3,i) = 0.0

         gradx(4,i) = 0.0
         grady(4,i) = 0.0
         gradz(4,i) = 0.0
 1000 continue

! If second order, loop over all the faces accumulate sums
!
!     nitfo = 0
!     if (ntt.gt.nitfo.or.ivisc.gt.0)then
!     if (1 .gt. 0) then
       do 1020 n = 1, nedge
         node1 = eptr(n,1)
         node2 = eptr(n,2)
!        if ((node1 .le. nnodes).or.(node2 .le. nnodes)) then
           dx1 = x(node2) - x(node1)
           dy1 = y(node2) - y(node1)
           dz1 = z(node2) - z(node1)
!
!          dist = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
!          weight = 1.0/dist
           weight = 1.0d0
!          w2 = weight*weight
!
           dx1 = weight*dx1
           dy1 = weight*dy1
           dz1 = weight*dz1

           flops = flops + 6.0
!
!         call PetscLogEventBegin(node1_event,0,0,0,0,ierr)
          if (node1 .le. nnodes) then
           dq1 = weight*(qnod(1,node2) - qnod(1,node1))
           dq2 = weight*(qnod(2,node2) - qnod(2,node1))
           dq3 = weight*(qnod(3,node2) - qnod(3,node1))
           dq4 = weight*(qnod(4,node2) - qnod(4,node1))
!
!          w11 = 1./(r11(node1)*r11(node1))
!          w22 = 1./(r22(node1)*r22(node1))
!          w33 = 1./(r33(node1)*r33(node1))
!          r12r11 = r12(node1)/r11(node1)
!          r13r11 = r13(node1)/r11(node1)
!          r23r22 = r23(node1)/r22(node1)
!          rmess  = (r12(node1)*r23(node1) - r13(node1)*r22(node1))/
!    1              (r11(node1)*r22(node1)*r33(node1)*r33(node1))
!
           w11 = r11(node1)
           r12r11 = r12(node1)
           r13r11 = r13(node1)
           w22 = r22(node1)
           r23r22 = r23(node1)
           w33 = r33(node1)
           rmess  = r44(node1)
!
           coef1  = dy1 - dx1*r12r11
           coef2  = dz1 - dx1*r13r11 - r23r22*coef1
           termx = dx1*w11 - w22*r12r11*coef1 + rmess*coef2
           termy = w22*coef1 - r23r22*w33*coef2
           termz = w33*coef2
!
           gradx(1,node1) = gradx(1,node1) + termx*dq1
           grady(1,node1) = grady(1,node1) + termy*dq1
           gradz(1,node1) = gradz(1,node1) + termz*dq1
!
           gradx(2,node1) = gradx(2,node1) + termx*dq2
           grady(2,node1) = grady(2,node1) + termy*dq2
           gradz(2,node1) = gradz(2,node1) + termz*dq2
!
           gradx(3,node1) = gradx(3,node1) + termx*dq3
           grady(3,node1) = grady(3,node1) + termy*dq3
           gradz(3,node1) = gradz(3,node1) + termz*dq3
!
           gradx(4,node1) = gradx(4,node1) + termx*dq4
           grady(4,node1) = grady(4,node1) + termy*dq4
           gradz(4,node1) = gradz(4,node1) + termz*dq4
!
           flops = flops + 49.0
          endif
!         call PetscLogEventEnd(node1_event,0,0,0,0,ierr)
!
! Now do the other node
!
!         call PetscLogEventBegin(node2_event,0,0,0,0,ierr)
          if (node2 .le. nnodes) then
           dx2 = -dx1
           dy2 = -dy1
           dz2 = -dz1
!
           dq1 = weight*(qnod(1,node1) - qnod(1,node2))
           dq2 = weight*(qnod(2,node1) - qnod(2,node2))
           dq3 = weight*(qnod(3,node1) - qnod(3,node2))
           dq4 = weight*(qnod(4,node1) - qnod(4,node2))
!
!          w11 = 1./(r11(node2)*r11(node2))
!          w22 = 1./(r22(node2)*r22(node2))
!          w33 = 1./(r33(node2)*r33(node2))
!          r12r11 = r12(node2)/r11(node2)
!          r13r11 = r13(node2)/r11(node2)
!          r23r22 = r23(node2)/r22(node2)
!          rmess  = (r12(node2)*r23(node2) - r13(node2)*r22(node2))/
!    &              (r11(node2)*r22(node2)*r33(node2)*r33(node2))

           w11 = r11(node2)
           r12r11 = r12(node2)
           r13r11 = r13(node2)
           w22 = r22(node2)
           r23r22 = r23(node2)
           w33 = r33(node2)
           rmess  = r44(node2)
!
           coef1  = dy2 - dx2*r12r11
           coef2  = dz2 - dx2*r13r11 - r23r22*coef1
           termx = dx2*w11 - w22*r12r11*coef1 + rmess*coef2
           termy = w22*coef1 - r23r22*w33*coef2
           termz = w33*coef2
!
           gradx(1,node2) = gradx(1,node2) + termx*dq1
           grady(1,node2) = grady(1,node2) + termy*dq1
           gradz(1,node2) = gradz(1,node2) + termz*dq1
!
           gradx(2,node2) = gradx(2,node2) + termx*dq2
           grady(2,node2) = grady(2,node2) + termy*dq2
           gradz(2,node2) = gradz(2,node2) + termz*dq2
!
           gradx(3,node2) = gradx(3,node2) + termx*dq3
           grady(3,node2) = grady(3,node2) + termy*dq3
           gradz(3,node2) = gradz(3,node2) + termz*dq3
!
           gradx(4,node2) = gradx(4,node2) + termx*dq4
           grady(4,node2) = grady(4,node2) + termy*dq4
           gradz(4,node2) = gradz(4,node2) + termz*dq4
!
           flops = flops + 52.0
          endif
!         call PetscLogEventEnd(node2_event,0,0,0,0,ierr)
!       endif
!
 1020  continue
!     end if
      call PetscLogFlops(flops,ierr)
!     call PetscLogEventEnd(grad_event,0,0,0,0,ierr)
!
! End of LSTGS
!
      return
      end

!=================================== GETRES ==========================72
!
! Calculates the residual
! Last Modified - D. K. Kaushik 1/23/97
! I have eliminated the input variables which were not needed -
! dq, A, B, iupdate
!
!=====================================================================72
#if defined(_OPENMP)
#if defined(HAVE_EDGE_COLORING)
      subroutine GETRES(nnodes,ncell,nedge,nsface,nvface,nfface,nbface,     &
     &                  nsnode,nvnode,nfnode,isface,ivface,ifface,          &
     &                  ileast,isnode,ivnode,ifnode,                        &
     &                  nnfacet,f2ntn,nnbound,                              &
     &                  nvfacet,f2ntv,nvbound,                              &
     &                  nffacet,f2ntf,nfbound,                              &
     &                  evec,sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,           &
     &                  xyzn,qvec,cdt,xyz,area,grad,resvec,turbre,          &
     &                  slen,c2n,c2e,us,vs,as,phi,amut,ires,                &
     &                  max_threads,                                        &
     &                  ncolor,ncount,                                      &
     &                  LocalTS,irank, nvertices)
#elif defined(HAVE_REDUNDANT_WORK)
      subroutine GETRES(nnodes,ncell,nedge,nsface,nvface,nfface,nbface,     &
     &                  nsnode,nvnode,nfnode,isface,ivface,ifface,          &
     &                  ileast,isnode,ivnode,ifnode,                        &
     &                  nnfacet,f2ntn,nnbound,                              &
     &                  nvfacet,f2ntv,nvbound,                              &
     &                  nffacet,f2ntf,nfbound,                              &
     &                  evec,sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,           &
     &                  xyzn,qvec,cdt,xyz,area,grad,resvec,turbre,          &
     &                  slen,c2n,c2e,us,vs,as,phi,amut,ires,                &
     &                  max_threads,                                        &
     &                  resd,                                               &
     &                  LocalTS,irank, nvertices)
#else
      subroutine GETRES(nnodes,ncell,nedge,nsface,nvface,nfface,nbface,     &
     &                  nsnode,nvnode,nfnode,isface,ivface,ifface,          &
     &                  ileast,isnode,ivnode,ifnode,                        &
     &                  nnfacet,f2ntn,nnbound,                              &
     &                  nvfacet,f2ntv,nvbound,                              &
     &                  nffacet,f2ntf,nfbound,                              &
     &                  evec,sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,           &
     &                  xyzn,qvec,cdt,xyz,area,grad,resvec,turbre,          &
     &                  slen,c2n,c2e,us,vs,as,phi,amut,ires,                &
     &                  max_threads,                                        &
     &                  nedgeAllThr,part_thr,nedge_thr,                     &
     &                  edge_thr,xyzn_thr,                                  &
     &                  LocalTS,irank, nvertices)
#endif
#else
      subroutine GETRES(nnodes,ncell,nedge,nsface,nvface,nfface,nbface,     &
     &                  nsnode,nvnode,nfnode,isface,ivface,ifface,          &
     &                  ileast,isnode,ivnode,ifnode,                        &
     &                  nnfacet,f2ntn,nnbound,                              &
     &                  nvfacet,f2ntv,nvbound,                              &
     &                  nffacet,f2ntf,nfbound,                              &
     &                  evec,sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,           &
     &                  xyzn,qvec,cdt,xyz,area,grad,resvec,turbre,          &
     &                  slen,c2n,c2e,us,vs,as,phi,amut,ires,                &
     &                  LocalTS,irank, nvertices)
#endif
!
      implicit none
#include <petsc/finclude/petscsys.h>
!
      integer nnodes, ncell, nedge,                                         &
     &        nbface,ileast,ires,                                           &
     &        nsface, nvface, nfface,                                       &
     &        nsnode, nvnode, nfnode,                                       &
     &        nnfacet,nvfacet,nffacet,                                      &
     &        nnbound,nvbound,nfbound,                                      &
     &        LocalTS,irank,nvertices
      integer evec(nedge,2)
      integer isface(1),ivface(1),ifface(1)
      integer isnode(1),ivnode(1),ifnode(1)
      integer c2n(ncell,4),c2e(ncell,6)
      integer f2ntn(nnfacet,4)
      integer f2ntv(nvfacet,4)
      integer f2ntf(nffacet,4)
!
      PetscScalar us(nbface,3,4),vs(nbface,3,4)
      PetscScalar as(nbface,3,4)
      PetscScalar sxn(1),syn(1),szn(1)
      PetscScalar vxn(1),vyn(1),vzn(1)
      PetscScalar fxn(1),fyn(1),fzn(1)
      PetscScalar xyz(nvertices,3),area(nvertices)
      PetscScalar xyzn(nedge,4)
      PetscScalar turbre(1),slen(1)
      PetscScalar qvec(nvertices,4)
      PetscScalar phi(nvertices,4)
      PetscScalar cdt(nvertices)
      PetscScalar grad(3,4,nvertices)
!     real dq(nnodes,4)
!     real r11(nvertices),r12(nvertices),r13(nvertices)
!     real r22(nvertices),r23(nvertices),r33(nvertices)
      PetscScalar amut(nnodes)
!
      integer i
      PetscScalar  title(20),beta,alpha
      PetscScalar  Re,dt,tot,res0,resc
      integer ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      PetscScalar  cfl1,cfl2
      integer nsmoth,iflim,itran,nbtran,jupdate,                             &
     &        nstage,ncyct,iramp,nitfo
      PetscScalar  gtol
      integer icycle,nsrch,ilu0,ifcn
      PetscScalar  rms(1001),clw(1001),cdw(1001),                                 &
     &        cmw(1001),xres(1001)
      common/info/title,beta,alpha,Re,dt,tot,res0,resc,                      &
     &            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,jupdate,              &
     &             nstage,ncyct,iramp,nitfo
      common/gmcom/gtol,icycle,nsrch,ilu0,ifcn
      common/history/rms,clw,cdw,cmw,xres
!     logging variables
!     integer flux_event, flag, ierr
!     integer delta2_event
!     character * 16 flux_label
!     character * 16 delta2_label
!     data flag/-1/,flux_label/'FluxEvaluation  '/
!     data delta2_label/'DELTA2          '/
!     save flux_event, flag, flux_label
!     save delta2_event,delta2_label
#if defined(INTERLACING)
       PetscScalar resvec(4,nnodes)
#undef res
#define res(i,j) resvec(i,j)
#else
       PetscScalar resvec(nnodes,4)
#define res(i,j) resvec(j,i)
#endif
#if defined(_OPENMP)
       integer max_threads
#if defined(HAVE_EDGE_COLORING)
       integer ncolor,ncount(1)
#elif defined(HAVE_REDUNDANT_WORK)
       PetscScalar resd(4,nnodes)
#else
       integer nedgeAllThr,part_thr(nnodes),nedge_thr(0:max_threads),         &
     &         edge_thr(2,nedgeAllThr)
       PetscScalar xyzn_thr(4,nedgeAllThr)
#endif
#endif
!

!     if (flag .eq. -1) then
!        call PetscLogEventRegister(delta2_event,delta2_label,ierr)
!        call PetscLogEventRegister(flux_event,flux_label,ierr)
!        flag = 1
!     endif
!
! Calculate the time step
!
      if (ires.eq.1) goto 888
!     call PetscLogEventBegin(delta2_event,0,0,0,0,ierr)
      call DELTAT2(nnodes,nedge,qvec,cdt,                                   &
     &             xyz,area,xyzn,evec,                                      &
     &             sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,                     &
     &             nsnode,nvnode,nfnode,isnode,ivnode,ifnode,               &
     &             LocalTS,irank,nvertices)
!     call PetscLogEventEnd(delta2_event,0,0,0,0,ierr)
  888 continue
!
!/*
!   Calculate the gradients
!   ----Kyle seems to recommend only LSTGS for gradients,
!   so I have commented the GETGRAD call - DKK (1/17/97)
!
!     if (ileast.eq.0) then
!        call GETGRAD(nnodes,ncell,nedge,nsface,nvface,nfface,
!    &                isface,ivface,ifface,eptr,ncolor,ncount,
!    &                qnod,gradx,grady,x,y,
!    &                area,wx,wy,xn,yn,rl)
!
!     else if (ileast.eq.4) then
!     if (ileast.eq.4) then
!        call LSTGS(nnodes,nedge,eptr,
!    &              qnod,gradx,grady,gradz,xyz,
!    &              r11,r12,r13,r22,r23,r33,irank,nvertices)
!     end if
!
! zero out residuals (viscous residuals are zeroed in vfluxnew)
!*/
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
!
!/*
!
! If not doing Newton-Krylov and iflim=1 call the Flux Limiter
!
!     if (iflim.eq.1.and.ifcn.ne.1)then
!      call TIMLIM(nnodes,nedge,qnod,res,dq,phi,ncolor,ncount,
!    &             gradx,grady,gradz,x,y,z,eptr)
!
! If we used the limiter we need to zero out the residual again
! since we used it for scratch space
!
!     do 1003 i = 1,nnodes
!         res(1,i)=0.0
!         res(2,i)=0.0
!         res(3,i)=0.0
!         res(4,i)=0.0
!1003   continue
!
!     end if
!*/
!   Split the fluxes and perform the flux balance
!
!       call PetscLogEventBegin(flux_event,0,0,0,0,ierr)

#if defined(_OPENMP)
#if defined(HAVE_EDGE_COLORING)
      call FLUX(nnodes, ncell, nedge,                                     &
     &          nsface, nvface, nfface, isface, ivface, ifface,           &
     &          nsnode, nvnode, nfnode, isnode, ivnode, ifnode,           &
     &          nnfacet,f2ntn,nnbound,                                    &
     &          nvfacet,f2ntv,nvbound,                                    &
     &          nffacet,f2ntf,nfbound,                                    &
     &          grad,evec,qvec,xyz,resvec,xyzn,                           &
     &          sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,phi,                  &
     &          max_threads,                                              &
     &          ncolor,ncount,                                            &
     &          irank,nvertices)
#elif defined(HAVE_REDUNDANT_WORK)
      call FLUX(nnodes, ncell, nedge,                                     &
     &          nsface, nvface, nfface, isface, ivface, ifface,           &
     &          nsnode, nvnode, nfnode, isnode, ivnode, ifnode,           &
     &          nnfacet,f2ntn,nnbound,                                    &
     &          nvfacet,f2ntv,nvbound,                                    &
     &          nffacet,f2ntf,nfbound,                                    &
     &          grad,evec,qvec,xyz,resvec,xyzn,                           &
     &          sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,phi,                  &
     &          max_threads,                                              &
     &          resd,                                                     &
     &          irank,nvertices)
#else
      call FLUX(nnodes, ncell, nedge,                                     &
     &          nsface, nvface, nfface, isface, ivface, ifface,           &
     &          nsnode, nvnode, nfnode, isnode, ivnode, ifnode,           &
     &          nnfacet,f2ntn,nnbound,                                    &
     &          nvfacet,f2ntv,nvbound,                                    &
     &          nffacet,f2ntf,nfbound,                                    &
     &          grad,evec,qvec,xyz,resvec,xyzn,                           &
     &          sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,phi,                  &
     &          max_threads,                                              &
     &          nedgeAllThr,part_thr,nedge_thr,edge_thr,xyzn_thr,         &
     &          irank,nvertices)
#endif
#else
      call FLUX(nnodes, ncell, nedge,                                     &
     &          nsface, nvface, nfface, isface, ivface, ifface,           &
     &          nsnode, nvnode, nfnode, isnode, ivnode, ifnode,           &
     &          nnfacet,f2ntn,nnbound,                                    &
     &          nvfacet,f2ntv,nvbound,                                    &
     &          nffacet,f2ntf,nfbound,                                    &
     &          grad,evec,qvec,xyz,resvec,xyzn,                           &
     &          sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,phi,                  &
     &          irank,nvertices)
#endif
!          call PetscLogEventEnd(flux_event,0,0,0,0,ierr)
!/*
! calculate viscous fluxes
!
!     if (ivisc.gt.0) then
!           call VISRHS (nnodes,ncell,nedge,
!           call EDGEVIS(nnodes,ncell,nedge,
!    &                  nsnode,nvnode,nfnode,isnode,ivnode,ifnode,
!    &                  nsface,nvface,nfface,isface,ivface,ifface,
!    &                  nnfacet,f2ntn,nnbound,ncolorn,countn,
!    &                  nvfacet,f2ntv,nvbound,ncolorv,countv,
!    &                  nffacet,f2ntf,nfbound,ncolorf,countf,
!    &                  nccolor,nccount,
!    &                  eptr,c2n,c2e,
!    &                  sxn,syn,szn,vxn,vyn,vzn,fxn,fyn,fzn,
!    &                  x,y,z,gradx,grady,gradz,
!    &                  qnod,amut,res,phi)
!     end if
!*/
! End of subroutine GETRES
!
      return
      end

!---------------------------------------------------------------
! The following subroutine is from node4t.f in the original
! code - D. K. Kaushik (1/17/97)
!---------------------------------------------------------------
!
!=============================================================================
!
!  Opens files for I/O
!
!=============================================================================
      SUBROUTINE OPENM(irank)
!
!  TAPE7  -- input:  mach number, angle of attack etc..
!  TAPE9  -- input:  reads restart file
!  TAPE10 -- output: residual history
!  TAPE11 -- output: writes restart file
!  TAPE12 -- output: writes residual and lift for plotting
!  TAPE13 -- output: writes flowfield for contour plotting
!
      implicit none
      integer  irank
!     OPEN(UNIT= 7,FILE='home/petsc/datafiles/fun3dgrid/ginput.faces',            &
!    &form='formatted',STATUS='OLD')

!     OPEN(UNIT=9,FILE='framer.bin',
!    &form='unformatted',STATUS='old')

      if (irank .eq. 0) OPEN(UNIT= 10,FILE='frame.out3',                          &
     &form='formatted',STATUS='unknown')

!     OPEN(UNIT= 11,FILE='frame.bin',
!    +form='unformatted',STATUS='unknown')

!     OPEN(UNIT= 12,FILE='frame.plt',
!    +form='formatted',STATUS='unknown')

!     OPEN(UNIT= 13,FILE='frame.tec',
!    +form='formatted',STATUS='unknown')

!     OPEN(UNIT= 14,FILE='frame.fast.g',
!    +form='unformatted',STATUS='unknown')

!     OPEN(UNIT= 15,FILE='frame.fast.q',
!    +form='unformatted',STATUS='unknown')

      return
      end
!
!
!===================================================================
!
! Get the IA, JA, and IAU arrays
!
!===================================================================
      subroutine GETIA(nnodes,nedge,evec,ia,ideg,irank)
      implicit none
#include <petsc/finclude/petscsys.h>
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
!
! First get the degree of each node using ideg as a dummy array
!
      do 1000 i = 1,nnodes
        ideg(i) = 0
 1000 continue
!
      do 1010 i = 1,nedge
        node1 = eptr(i,1)
        node2 = eptr(i,2)
#if defined(_OPENMP) && !defined(HAVE_REDUNDANT_WORK) && !defined(HAVE_EDGE_COLORING)
        ideg(node1) = ideg(node1) + 1
        ideg(node2) = ideg(node2) + 1
#else
        if (node1 .le. nnodes) ideg(node1) = ideg(node1) + 1
        if (node2 .le. nnodes) ideg(node2) = ideg(node2) + 1
#endif
 1010 continue
!
! Now we can fill the ia array fairly easily
!
      ia(1) = 1
      do 1020 i = 1,nnodes
        ia(i+1) = ia(i) + ideg(i) + 1
!       write(9,100)i,ideg(i)
! 100   format(1h ,'deg(',i6,')=',i6)
 1020 continue
!
      return
      end
!
!===================================================================
!
! Get the IA, JA, and IAU arrays
!
!===================================================================
      subroutine GETJA(nnodes,nedge,evec,ia,ja,iwork,irank)
      implicit none
#include <petsc/finclude/petscsys.h>
      integer nnodes,nedge,irank
      integer ia(1),ja(1),iwork(1)
#if defined(INTERLACING)
       integer evec(2,nedge)
#define eptr(j,i) evec(i,j)
#else
       integer evec(nedge,2)
#define eptr(i,j) evec(i,j)
#endif
       integer i,index,node1,node2,index1,index2,                          &
     &         istart,iend
!     open(unit=90,file='map.out',status='UNKNOWN')
!
! Now we need to get the JA array
! First fill the diagonal places
!
      do 1040 i = 1,nnodes
        index = ia(i)
        ja(index) = i
        iwork(i) = 1
 1040 continue
!
      do 1030 i = 1,nedge
        node1 = eptr(i,1)
        node2 = eptr(i,2)
!
#if defined(_OPENMP) && !defined(HAVE_REDUNDANT_WORK) && !defined(HAVE_EDGE_COLORING)
        index1 = ia(node1) + iwork(node1)
        iwork(node1) = iwork(node1) + 1
        ja(index1) = node2
        index2 = ia(node2) + iwork(node2)
        iwork(node2) = iwork(node2) + 1
        ja(index2) = node1
#else
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
#endif
 1030 continue
!
! Now lets sort all our "bins" and get the correct one on the diagonal
!
      do 1050 i = 1,nnodes
        istart = ia(i)
        iend   = ia(i+1) - 1
!       write(9,200)i,istart,iend
! 200   format(1h ,'Sorting ',i6,' istart iend = ',i6,1x,i6)
        call SORTER(istart,iend,ja,i)
 1050 continue
!
! Now get the "fhelp" array which will assist in assembling
! the flux Jacobians into the correct location in the alu array
!
!     write(90,*) 'fhelp array'
!     do 1060 i = 1,nedge
!       node1 = eptr(i,1)
!       node2 = eptr(i,2)
!
! First take care of node1
!
!       idiag = iau(node1)
!
! If the offdiagonal term is ordered later in the ja array
!
!       if (node2.gt.node1)then
!        jstart = idiag + 1
!        jend   = ia(node1+1) - 1
!       else
!        jstart = ia(node1)
!        jend   = idiag -1
!       end if
!
!        do 1070 j = jstart,jend
!          if (ja(j).eq.node2)fhelp(i,1) = j
!1070    continue
!
!
! Now take care of node2
!
!       idiag = iau(node2)
!
! If the offdiagonal term is ordered later in the ja array
!
!       if (node1.gt.node2)then
!        jstart = idiag + 1
!        jend   = ia(node2+1) - 1
!       else
!        jstart = ia(node2)
!        jend   = idiag -1
!       end if
!
!       do 1080 j = jstart,jend
!         if (ja(j).eq.node1)fhelp(i,2) = j
!1080   continue
!       write(90,*) i,fhelp(i,1),fhelp(i,2)
!1060 continue
!      close(90)
!
      return
      end
!
!
!===================================================================
!
! Sort each of our bins
!
!===================================================================
      subroutine SORTER(istart,iend,ja,inode)
      implicit none
#include <petsc/finclude/petscsys.h>
      integer istart,iend,inode
      integer ja(1)
!
      integer min,minsave,jsave,i,j
      do 1000 i = istart,iend
        min = ja(i)
        minsave = ja(i)
        jsave = i
        do 1010 j = i+1,iend
          if (ja(j).lt.min)then
            min = ja(j)
            jsave = j
          end if
 1010   continue
        ja(i) = min
        ja(jsave) = minsave
!       if (ja(i).eq.inode)iau(inode) = i
 1000 continue
!
      return
      end

