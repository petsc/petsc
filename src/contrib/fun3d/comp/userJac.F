!---------------------------------------------------------------
! The following subroutines are from node6t.f in the original
! code.
! Last Modified - D. K. Kaushik (1/24/97)
!---------------------------------------------------------------
!
!=================================== FILLA ===================================
!
! Fill the nonzero term of the A matrix
!   - viscous terms are not included right now
!
!  New version put in by Eric for David Keyes' stuff
!   Includes complete "by hand" differentiation
! Modified - D. K. Kaushik (1/9/98)
! cfl is passed as a parameter
!
!=============================================================================
      subroutine FILLA(nnodes,nedge,evec,                                       &
     &                 nsface,isface,fxn,fyn,fzn,sxn,syn,szn,                   &
     &                 nsnode,nvnode,nfnode,isnode,ivnode,ifnode,               &
     &                 qvec,A,cdt,                                              &
     &                 rl,vol,xn,yn,zn,cfl,irank,nvertices)
!
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscmat.h>
#include <petsc/finclude/petscpc.h>
#include <petsc/finclude/petscsnes.h>
!
      common/runge/cfl1,cfl2,nsmoth,iflim,itran,nbtran,ipv,
     &             nstage,ncyct,iramp,nitfo,ncyc
      common/info/title(20),xmach,alpha,yaw,Re,dt,tot,res0,resc,
     1            ntt,mseq,ivisc,irest,icyc,ihane,ntturb
      common/fluid/gamma,gm1,gp1,gm1g,gp1g,ggm1
      common/ivals/p0,rho0,c0,u0,v0,w0,et0,h0,pt0
      common/gmcom/gtol,icycle,nsrch,ilu0,ifcn
!
      real xn(nedge),yn(nedge),zn(nedge),rl(nedge),vol(1)
      real sxn(1),syn(1),szn(1)
      real fxn(1),fyn(1),fzn(1), pio, cio, pbo
      real cdt(1)
      real val(50), val1(10,5)
!     real A(nnz,4,4)
      Mat  A
      MatStructure flag


      integer isface(1)
      integer isnode(1),ivnode(1),ifnode(1)
!     integer ia(1),ja(1),iau(1),fhelp(nedge,2)
      integer  irow(25), icol(25)
      integer i, j, k, in
      PetscLogDouble flops
!
#if defined(INTERLACING)
      real qvec(5,nvertices)
      integer evec(2,nedge)
#define qnode(i,j) qvec(i,j)
#define dfp(a,b) val1(b,a)
#define dfm(a,b) val1(b+5,a)
#define eptr(j,i) evec(i,j)
#else
      real qvec(nvertices,5)
      integer evec(nedge,2)
#define qnode(i,j) qvec(j,i)
#define dfp(a,b) val1(b,a)
#define dfm(a,b) val1(b+5,a)
#define eptr(i,j) evec(i,j)
#endif
!
! Loop over the edges and fill the matrix
! First just zero it out
!
      flops = 0.0
      call MatZeroEntries(A,ierr)
!     do 1000 j = 1,4
!       do 1000 k = 1,4
!         do 1000 i = 1,nnz
!           A(i,j,k) = 0.0
!1000 continue
!     print *, "I am in FILLA"
!
! Take care of the time step term on the diagonal
!
!     cfl = abs(dt)
!     if (iramp.lt.0) then
!       cfl = cfl1*res0/resc
!       if (cfl.gt.cfl2) cfl=cfl2
!       if (ntt.eq.1.and.irest.eq.0) cfl=10
!     else
!      if (dt.lt.0.0) cfl = cfl1
!      if (dt.lt.0.0.and.iramp.gt.0.and.ntt.gt.iramp) cfl = cfl2
!      if (dt.lt.0.0.and.iramp.gt.0.and.ntt.le.iramp) then
!       cfl = cfl1 + (cfl2 - cfl1)*(ntt - 1)/iramp
!      end if
!     end if
!
!       write(6,555)res0,resc,cfl,cfl1,cfl2
!
#if defined(INTERLACING)
      do i = 1,nnodes
       temp = vol(i)/(cfl*cdt(i))
       do j = 1,5
        ir = j - 1 + 5*(i-1)
        call MatSetValuesLocal(A,1,ir,1,ir,temp,ADD_VALUES,ierr)
       enddo
      enddo
      flops = flops + 2*nnodes
#else
      do j = 1,5
       do i = 1,nnodes
        temp = vol(i)/(cfl*cdt(i))
        ir = i - 1 + nnodes*(j-1)
        call MatSetValues(A,1,ir,1,ir,temp,ADD_VALUES,ierr)
      enddo
      enddo
      flops = flops + 5*2*nnodes
#endif

!     call MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY,ierr)
!     call MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY,ierr)
!     print *, "Finished doing time stepping part to diagonal term"
!
! Now fill A from interior contributions
! We will fix the boundaries later
!
!  First convert to primitive variables
!
      call ETOP(nvertices,qvec)
!
      vconst = 1.0
      if (ivisc.eq.0) vconst = 0.0
      xmr = xmach/Re
      c512 = 5./12.*xmr*vconst
      flops = flops + 4.0
!
      do 1030 n = 1, nedge
       node1 = eptr(n,1)
       node2 = eptr(n,2)
       if ((node1 .le. nnodes).or.(node2 .le. nnodes)) then
!
! Get unit normals and area
!
          xnorm = xn(n)
          ynorm = yn(n)
          znorm = zn(n)
          area  = rl(n)
!
! Get variables on "left" side of face
!
            rhol   = qnode(1,node1)
              rholrL = 1.
              rholuL = 0.
              rholvL = 0.
              rholwL = 0.
              rholpL = 0.
!
              rholrR = 0.
              rholuR = 0.
              rholvR = 0.
              rholwR = 0.
              rholpR = 0.
            ul     = qnode(2,node1)
              ulrL = 0.0
              uluL = 1.0
              ulvL = 0.0
              ulwL = 0.0
              ulpL = 0.0
!
              ulrR = 0.0
              uluR = 0.0
              ulvR = 0.0
              ulwR = 0.0
              ulpR = 0.0
            vl     = qnode(3,node1)
              vlrL = 0.0
              vluL = 0.0
              vlvL = 1.0
              vlwL = 0.0
              vlpL = 0.0
!
              vlrR = 0.0
              vluR = 0.0
              vlvR = 0.0
              vlwR = 0.0
              vlpR = 0.0
            wl     = qnode(4,node1)
              wlrL = 0.0
              wluL = 0.0
              wlvL = 0.0
              wlwL = 1.0
              wlpL = 0.0
!
              wlrR = 0.0
              wluR = 0.0
              wlvR = 0.0
              wlwR = 0.0
              wlpR = 0.0
!
            q2l     = ul*ul + vl*vl + wl*wl
              q2lrL = 2.*ul*ulrL + 2.*vl*vlrL + 2.*wl*wlrL
              q2luL = 2.*ul*uluL + 2.*vl*vluL + 2.*wl*wluL
              q2lvL = 2.*ul*ulvL + 2.*vl*vlvL + 2.*wl*wlvL
              q2lwL = 2.*ul*ulwL + 2.*vl*vlwL + 2.*wl*wlwL
              q2lpL = 2.*ul*ulpL + 2.*vl*vlpL + 2.*wl*wlpL
!
              q2lrR = 2.*ul*ulrR + 2.*vl*vlrR + 2.*wl*wlrR
              q2luR = 2.*ul*uluR + 2.*vl*vluR + 2.*wl*wluR
              q2lvR = 2.*ul*ulvR + 2.*vl*vlvR + 2.*wl*wlvR
              q2lwR = 2.*ul*ulwR + 2.*vl*vlwR + 2.*wl*wlwR
              q2lpR = 2.*ul*ulpR + 2.*vl*vlpR + 2.*wl*wlpR
!
            pressl = qnode(5,node1)
              presslrL = 0.
              pressluL = 0.
              presslvL = 0.
              presslwL = 0.
              presslpL = 1.
!
              presslrR = 0.
              pressluR = 0.
              presslvR = 0.
              presslwR = 0.
              presslpR = 0.
!
            enrgyl = pressl/gm1 + .5*rhol*q2l
!
              enrgylrL = presslrL/gm1 + .5*(rhol*q2lrL + q2l*rholrL)
              enrgyluL = pressluL/gm1 + .5*(rhol*q2luL + q2l*rholuL)
              enrgylvL = presslvL/gm1 + .5*(rhol*q2lvL + q2l*rholvL)
              enrgylwL = presslwL/gm1 + .5*(rhol*q2lwL + q2l*rholwL)
              enrgylpL = presslpL/gm1 + .5*(rhol*q2lpL + q2l*rholpL)
!
              enrgylrR = presslrR/gm1 + .5*(rhol*q2lrR + q2l*rholrR)
              enrgyluR = pressluR/gm1 + .5*(rhol*q2luR + q2l*rholuR)
              enrgylvR = presslvR/gm1 + .5*(rhol*q2lvR + q2l*rholvR)
              enrgylwR = presslwR/gm1 + .5*(rhol*q2lwR + q2l*rholwR)
              enrgylpR = presslpR/gm1 + .5*(rhol*q2lpR + q2l*rholpR)
!
            Hl     = (enrgyl + pressl)/rhol
!
            HlrL = (rhol*(enrgylrL+presslrL) - (enrgyl+pressl)*rholrL)
     &             / rhol / rhol
            HluL = (rhol*(enrgyluL+pressluL) - (enrgyl+pressl)*rholuL)
     &             / rhol / rhol
            HlvL = (rhol*(enrgylvL+presslvL) - (enrgyl+pressl)*rholvL)
     &             / rhol / rhol
            HlwL = (rhol*(enrgylwL+presslwL) - (enrgyl+pressl)*rholwL)
     &             / rhol / rhol
            HlpL = (rhol*(enrgylpL+presslpL) - (enrgyl+pressl)*rholpL)
     &             / rhol / rhol
!
            HlrR = (rhol*(enrgylrR+presslrR) - (enrgyl+pressl)*rholrR)
     &             / rhol / rhol
            HluR = (rhol*(enrgyluR+pressluR) - (enrgyl+pressl)*rholuR)
     &             / rhol / rhol
            HlvR = (rhol*(enrgylvR+presslvR) - (enrgyl+pressl)*rholvR)
     &             / rhol / rhol
            HlwR = (rhol*(enrgylwR+presslwR) - (enrgyl+pressl)*rholwR)
     &             / rhol / rhol
            HlpR = (rhol*(enrgylpR+presslpR) - (enrgyl+pressl)*rholpR)
     &             / rhol / rhol
!
!
            cl     = sqrt(gamma*pressl/rhol)
!
            clrL = 0.5 / sqrt(gamma*pressl/rhol) *
     &             gamma * (rhol*presslrL - pressl*rholrL)
     &             / rhol / rhol
            cluL = 0.5 / sqrt(gamma*pressl/rhol) *
     &             gamma * (rhol*pressluL - pressl*rholuL)
     &             / rhol / rhol
            clvL = 0.5 / sqrt(gamma*pressl/rhol) *
     &             gamma * (rhol*presslvL - pressl*rholvL)
     &             / rhol / rhol
            clwL = 0.5 / sqrt(gamma*pressl/rhol) *
     &             gamma * (rhol*presslwL - pressl*rholwL)
     &             / rhol / rhol
            clpL = 0.5 / sqrt(gamma*pressl/rhol) *
     &             gamma * (rhol*presslpL - pressl*rholpL)
     &             / rhol / rhol
!
            clrR = 0.5 / sqrt(gamma*pressl/rhol) *
     &             gamma * (rhol*presslrR - pressl*rholrR)
     &             / rhol / rhol
            cluR = 0.5 / sqrt(gamma*pressl/rhol) *
     &             gamma * (rhol*pressluR - pressl*rholuR)
     &             / rhol / rhol
            clvR = 0.5 / sqrt(gamma*pressl/rhol) *
     &             gamma * (rhol*presslvR - pressl*rholvR)
     &             / rhol / rhol
            clwR = 0.5 / sqrt(gamma*pressl/rhol) *
     &             gamma * (rhol*presslwR - pressl*rholwR)
     &             / rhol / rhol
            clpR = 0.5 / sqrt(gamma*pressl/rhol) *
     &             gamma * (rhol*presslpR - pressl*rholpR)
     &             / rhol / rhol
!
!
            ubarl  = xnorm*ul + ynorm*vl + znorm*wl
!
            ubarlrL = xnorm*ulrL + ynorm*vlrL + znorm*wlrL
            ubarluL = xnorm*uluL + ynorm*vluL + znorm*wluL
            ubarlvL = xnorm*ulvL + ynorm*vlvL + znorm*wlvL
            ubarlwL = xnorm*ulwL + ynorm*vlwL + znorm*wlwL
            ubarlpL = xnorm*ulpL + ynorm*vlpL + znorm*wlpL
!
            ubarlrR = xnorm*ulrR + ynorm*vlrR + znorm*wlrR
            ubarluR = xnorm*uluR + ynorm*vluR + znorm*wluR
            ubarlvR = xnorm*ulvR + ynorm*vlvR + znorm*wlvR
            ubarlwR = xnorm*ulwR + ynorm*vlwR + znorm*wlwR
            ubarlpR = xnorm*ulpR + ynorm*vlpR + znorm*wlpR
!
! Get variables on "right" side of face
!
            rhor   = qnode(1,node2)
              rhorrL = 0.
              rhoruL = 0.
              rhorvL = 0.
              rhorwL = 0.
              rhorpL = 0.
!
              rhorrR = 1.
              rhoruR = 0.
              rhorvR = 0.
              rhorwR = 0.
              rhorpR = 0.
!
            ur     = qnode(2,node2)
              urrL = 0.0
              uruL = 0.0
              urvL = 0.0
              urwL = 0.0
              urpL = 0.0
!
              urrR = 0.0
              uruR = 1.0
              urvR = 0.0
              urwR = 0.0
              urpR = 0.0
!
            vr     = qnode(3,node2)
              vrrL = 0.0
              vruL = 0.0
              vrvL = 0.0
              vrwL = 0.0
              vrpL = 0.0
!
              vrrR = 0.0
              vruR = 0.0
              vrvR = 1.0
              vrwR = 0.0
              vrpR = 0.0
!
            wr     = qnode(4,node2)
              wrrL = 0.0
              wruL = 0.0
              wrvL = 0.0
              wrwL = 0.0
              wrpL = 0.0
!
              wrrR = 0.0
              wruR = 0.0
              wrvR = 0.0
              wrwR = 1.0
              wrpR = 0.0
!
            q2r    = ur*ur + vr*vr + wr*wr
              q2rrL = 2.*ur*urrL + 2.*vr*vrrL + 2.*wr*wrrL
              q2ruL = 2.*ur*uruL + 2.*vr*vruL + 2.*wr*wruL
              q2rvL = 2.*ur*urvL + 2.*vr*vrvL + 2.*wr*wrvL
              q2rwL = 2.*ur*urwL + 2.*vr*vrwL + 2.*wr*wrwL
              q2rpL = 2.*ur*urpL + 2.*vr*vrpL + 2.*wr*wrpL
!
              q2rrR = 2.*ur*urrR + 2.*vr*vrrR + 2.*wr*wrrR
              q2ruR = 2.*ur*uruR + 2.*vr*vruR + 2.*wr*wruR
              q2rvR = 2.*ur*urvR + 2.*vr*vrvR + 2.*wr*wrvR
              q2rwR = 2.*ur*urwR + 2.*vr*vrwR + 2.*wr*wrwR
              q2rpR = 2.*ur*urpR + 2.*vr*vrpR + 2.*wr*wrpR
!
            pressr = qnode(5,node2)
              pressrrL = 0.
              pressruL = 0.
              pressrvL = 0.
              pressrwL = 0.
              pressrpL = 0.
!
              pressrrR = 0.
              pressruR = 0.
              pressrvR = 0.
              pressrwR = 0.
              pressrpR = 1.
!
            enrgyr = pressr/gm1 + .5*rhor*q2r
!
              enrgyrrL = pressrrL/gm1 + .5*(rhor*q2rrL + q2r*rhorrL)
              enrgyruL = pressruL/gm1 + .5*(rhor*q2ruL + q2r*rhoruL)
              enrgyrvL = pressrvL/gm1 + .5*(rhor*q2rvL + q2r*rhorvL)
              enrgyrwL = pressrwL/gm1 + .5*(rhor*q2rwL + q2r*rhorwL)
              enrgyrpL = pressrpL/gm1 + .5*(rhor*q2rpL + q2r*rhorpL)
!
              enrgyrrR = pressrrR/gm1 + .5*(rhor*q2rrR + q2r*rhorrR)
              enrgyruR = pressruR/gm1 + .5*(rhor*q2ruR + q2r*rhoruR)
              enrgyrvR = pressrvR/gm1 + .5*(rhor*q2rvR + q2r*rhorvR)
              enrgyrwR = pressrwR/gm1 + .5*(rhor*q2rwR + q2r*rhorwR)
              enrgyrpR = pressrpR/gm1 + .5*(rhor*q2rpR + q2r*rhorpR)
!
            Hr     = (enrgyr + pressr)/rhor
!
            HrrL = (rhor*(enrgyrrL+pressrrL) - (enrgyr+pressr)*rhorrL)
     &             / rhor / rhor
            HruL = (rhor*(enrgyruL+pressruL) - (enrgyr+pressr)*rhoruL)
     &             / rhor / rhor
            HrvL = (rhor*(enrgyrvL+pressrvL) - (enrgyr+pressr)*rhorvL)
     &             / rhor / rhor
            HrwL = (rhor*(enrgyrwL+pressrwL) - (enrgyr+pressr)*rhorwL)
     &             / rhor / rhor
            HrpL = (rhor*(enrgyrpL+pressrpL) - (enrgyr+pressr)*rhorpL)
     &             / rhor / rhor
!
            HrrR = (rhor*(enrgyrrR+pressrrR) - (enrgyr+pressr)*rhorrR)
     &             / rhor / rhor
            HruR = (rhor*(enrgyruR+pressruR) - (enrgyr+pressr)*rhoruR)
     &             / rhor / rhor
            HrvR = (rhor*(enrgyrvR+pressrvR) - (enrgyr+pressr)*rhorvR)
     &             / rhor / rhor
            HrwR = (rhor*(enrgyrwR+pressrwR) - (enrgyr+pressr)*rhorwR)
     &             / rhor / rhor
            HrpR = (rhor*(enrgyrpR+pressrpR) - (enrgyr+pressr)*rhorpR)
     &             / rhor / rhor
!
!
            cr     = sqrt(gamma*pressr/rhor)
!
            crrL = 0.5 / sqrt(gamma*pressr/rhor) *
     &             gamma * (rhor*pressrrL - pressr*rhorrL)
     &             / rhor / rhor
            cruL = 0.5 / sqrt(gamma*pressr/rhor) *
     &             gamma * (rhor*pressruL - pressr*rhoruL)
     &             / rhor / rhor
            crvL = 0.5 / sqrt(gamma*pressr/rhor) *
     &             gamma * (rhor*pressrvL - pressr*rhorvL)
     &             / rhor / rhor
            crwL = 0.5 / sqrt(gamma*pressr/rhor) *
     &             gamma * (rhor*pressrwL - pressr*rhorwL)
     &             / rhor / rhor
            crpL = 0.5 / sqrt(gamma*pressr/rhor) *
     &             gamma * (rhor*pressrpL - pressr*rhorpL)
     &             / rhor / rhor
!
            crrR = 0.5 / sqrt(gamma*pressr/rhor) *
     &             gamma * (rhor*pressrrR - pressr*rhorrR)
     &             / rhor / rhor
            cruR = 0.5 / sqrt(gamma*pressr/rhor) *
     &             gamma * (rhor*pressruR - pressr*rhoruR)
     &             / rhor / rhor
            crvR = 0.5 / sqrt(gamma*pressr/rhor) *
     &             gamma * (rhor*pressrvR - pressr*rhorvR)
     &             / rhor / rhor
            crwR = 0.5 / sqrt(gamma*pressr/rhor) *
     &             gamma * (rhor*pressrwR - pressr*rhorwR)
     &             / rhor / rhor
            crpR = 0.5 / sqrt(gamma*pressr/rhor) *
     &             gamma * (rhor*pressrpR - pressr*rhorpR)
     &             / rhor / rhor
!
!
            ubarr  = xnorm*ur + ynorm*vr + znorm*wr
!
            ubarrrL = xnorm*urrL + ynorm*vrrL + znorm*wrrL
            ubarruL = xnorm*uruL + ynorm*vruL + znorm*wruL
            ubarrvL = xnorm*urvL + ynorm*vrvL + znorm*wrvL
            ubarrwL = xnorm*urwL + ynorm*vrwL + znorm*wrwL
            ubarrpL = xnorm*urpL + ynorm*vrpL + znorm*wrpL
!
            ubarrrR = xnorm*urrR + ynorm*vrrR + znorm*wrrR
            ubarruR = xnorm*uruR + ynorm*vruR + znorm*wruR
            ubarrvR = xnorm*urvR + ynorm*vrvR + znorm*wrvR
            ubarrwR = xnorm*urwR + ynorm*vrwR + znorm*wrwR
            ubarrpR = xnorm*urpR + ynorm*vrpR + znorm*wrpR
!
! Compute rho averages
!
            rho = sqrt(rhol*rhor)
!
            rhorL = 0.5 / sqrt(rhol*rhor) * (rhol*rhorrL +
     &              rhor*rholrL)
            rhouL = 0.5 / sqrt(rhol*rhor) * (rhol*rhoruL +
     &              rhor*rholuL)
            rhovL = 0.5 / sqrt(rhol*rhor) * (rhol*rhorvL +
     &              rhor*rholvL)
            rhowL = 0.5 / sqrt(rhol*rhor) * (rhol*rhorwL +
     &              rhor*rholwL)
            rhopL = 0.5 / sqrt(rhol*rhor) * (rhol*rhorpL +
     &              rhor*rholpL)
!
            rhorR = 0.5 / sqrt(rhol*rhor) * (rhol*rhorrR +
     &              rhor*rholrR)
            rhouR = 0.5 / sqrt(rhol*rhor) * (rhol*rhoruR +
     &              rhor*rholuR)
            rhovR = 0.5 / sqrt(rhol*rhor) * (rhol*rhorvR +
     &              rhor*rholvR)
            rhowR = 0.5 / sqrt(rhol*rhor) * (rhol*rhorwR +
     &              rhor*rholwR)
            rhopR = 0.5 / sqrt(rhol*rhor) * (rhol*rhorpR +
     &              rhor*rholpR)
!
            wat = rho/(rho + rhor)
!
            watrL = ((rho + rhor)*rhorL - rho*(rhorL + rhorrL)) /
     &              (rho + rhor) / (rho + rhor)
            watuL = ((rho + rhor)*rhouL - rho*(rhouL + rhoruL)) /
     &              (rho + rhor) / (rho + rhor)
            watvL = ((rho + rhor)*rhovL - rho*(rhovL + rhorvL)) /
     &              (rho + rhor) / (rho + rhor)
            watwL = ((rho + rhor)*rhowL - rho*(rhowL + rhorwL)) /
     &              (rho + rhor) / (rho + rhor)
            watpL = ((rho + rhor)*rhopL - rho*(rhopL + rhorpL)) /
     &              (rho + rhor) / (rho + rhor)
!
            watrR = ((rho + rhor)*rhorR - rho*(rhorR + rhorrR)) /
     &              (rho + rhor) / (rho + rhor)
            watuR = ((rho + rhor)*rhouR - rho*(rhouR + rhoruR)) /
     &              (rho + rhor) / (rho + rhor)
            watvR = ((rho + rhor)*rhovR - rho*(rhovR + rhorvR)) /
     &              (rho + rhor) / (rho + rhor)
            watwR = ((rho + rhor)*rhowR - rho*(rhowR + rhorwR)) /
     &              (rho + rhor) / (rho + rhor)
            watpR = ((rho + rhor)*rhopR - rho*(rhopR + rhorpR)) /
     &              (rho + rhor) / (rho + rhor)
!
!234567890c234567890c234567890c234567890c234567890c234567890c23456789012
!
            u   = ul*wat + ur*(1. - wat)
!
            urL = ul*watrL + wat*ulrL + ur*(-watrL) + (1. - wat)*urrL
            uuL = ul*watuL + wat*uluL + ur*(-watuL) + (1. - wat)*uruL
            uvL = ul*watvL + wat*ulvL + ur*(-watvL) + (1. - wat)*urvL
            uwL = ul*watwL + wat*ulwL + ur*(-watwL) + (1. - wat)*urwL
            upL = ul*watpL + wat*ulpL + ur*(-watpL) + (1. - wat)*urpL
!
            urR = ul*watrR + wat*ulrR + ur*(-watrR) + (1. - wat)*urrR
            uuR = ul*watuR + wat*uluR + ur*(-watuR) + (1. - wat)*uruR
            uvR = ul*watvR + wat*ulvR + ur*(-watvR) + (1. - wat)*urvR
            uwR = ul*watwR + wat*ulwR + ur*(-watwR) + (1. - wat)*urwR
            upR = ul*watpR + wat*ulpR + ur*(-watpR) + (1. - wat)*urpR
!
            v   = vl*wat + vr*(1. - wat)
!
            vrL = vl*watrL + wat*vlrL + vr*(-watrL) + (1. - wat)*vrrL
            vuL = vl*watuL + wat*vluL + vr*(-watuL) + (1. - wat)*vruL
            vvL = vl*watvL + wat*vlvL + vr*(-watvL) + (1. - wat)*vrvL
            vwL = vl*watwL + wat*vlwL + vr*(-watwL) + (1. - wat)*vrwL
            vpL = vl*watpL + wat*vlpL + vr*(-watpL) + (1. - wat)*vrpL
!
            vrR = vl*watrR + wat*vlrR + vr*(-watrR) + (1. - wat)*vrrR
            vuR = vl*watuR + wat*vluR + vr*(-watuR) + (1. - wat)*vruR
            vvR = vl*watvR + wat*vlvR + vr*(-watvR) + (1. - wat)*vrvR
            vwR = vl*watwR + wat*vlwR + vr*(-watwR) + (1. - wat)*vrwR
            vpR = vl*watpR + wat*vlpR + vr*(-watpR) + (1. - wat)*vrpR
!
            w   = wl*wat + wr*(1. - wat)
!
            wrL = wl*watrL + wat*wlrL + wr*(-watrL) + (1. - wat)*wrrL
            wuL = wl*watuL + wat*wluL + wr*(-watuL) + (1. - wat)*wruL
            wvL = wl*watvL + wat*wlvL + wr*(-watvL) + (1. - wat)*wrvL
            wwL = wl*watwL + wat*wlwL + wr*(-watwL) + (1. - wat)*wrwL
            wpL = wl*watpL + wat*wlpL + wr*(-watpL) + (1. - wat)*wrpL
!
            wrR = wl*watrR + wat*wlrR + wr*(-watrR) + (1. - wat)*wrrR
            wuR = wl*watuR + wat*wluR + wr*(-watuR) + (1. - wat)*wruR
            wvR = wl*watvR + wat*wlvR + wr*(-watvR) + (1. - wat)*wrvR
            wwR = wl*watwR + wat*wlwR + wr*(-watwR) + (1. - wat)*wrwR
            wpR = wl*watpR + wat*wlpR + wr*(-watpR) + (1. - wat)*wrpR
!
            H   = Hl*wat + Hr*(1. - wat)
!
            HrL = Hl*watrL + wat*HlrL + Hr*(-watrL) + (1. - wat)*HrrL
            HuL = Hl*watuL + wat*HluL + Hr*(-watuL) + (1. - wat)*HruL
            HvL = Hl*watvL + wat*HlvL + Hr*(-watvL) + (1. - wat)*HrvL
            HwL = Hl*watwL + wat*HlwL + Hr*(-watwL) + (1. - wat)*HrwL
            HpL = Hl*watpL + wat*HlpL + Hr*(-watpL) + (1. - wat)*HrpL
!
            HrR = Hl*watrR + wat*HlrR + Hr*(-watrR) + (1. - wat)*HrrR
            HuR = Hl*watuR + wat*HluR + Hr*(-watuR) + (1. - wat)*HruR
            HvR = Hl*watvR + wat*HlvR + Hr*(-watvR) + (1. - wat)*HrvR
            HwR = Hl*watwR + wat*HlwR + Hr*(-watwR) + (1. - wat)*HrwR
            HpR = Hl*watpR + wat*HlpR + Hr*(-watpR) + (1. - wat)*HrpR
!
            q2  = u*u + v*v + w*w
!
            q2rL = 2.*u*urL + 2.*v*vrL + 2.*w*wrL
            q2uL = 2.*u*uuL + 2.*v*vuL + 2.*w*wuL
            q2vL = 2.*u*uvL + 2.*v*vvL + 2.*w*wvL
            q2wL = 2.*u*uwL + 2.*v*vwL + 2.*w*wwL
            q2pL = 2.*u*upL + 2.*v*vpL + 2.*w*wpL
!
            q2rR = 2.*u*urR + 2.*v*vrR + 2.*w*wrR
            q2uR = 2.*u*uuR + 2.*v*vuR + 2.*w*wuR
            q2vR = 2.*u*uvR + 2.*v*vvR + 2.*w*wvR
            q2wR = 2.*u*uwR + 2.*v*vwR + 2.*w*wwR
            q2pR = 2.*u*upR + 2.*v*vpR + 2.*w*wpR
!
            c   = sqrt(gm1*(H - 0.5*q2))
!
            crL = 0.5 / c * gm1*(HrL - 0.5*q2rL)
            cuL = 0.5 / c * gm1*(HuL - 0.5*q2uL)
            cvL = 0.5 / c * gm1*(HvL - 0.5*q2vL)
            cwL = 0.5 / c * gm1*(HwL - 0.5*q2wL)
            cpL = 0.5 / c * gm1*(HpL - 0.5*q2pL)
!
            crR = 0.5 / c * gm1*(HrR - 0.5*q2rR)
            cuR = 0.5 / c * gm1*(HuR - 0.5*q2uR)
            cvR = 0.5 / c * gm1*(HvR - 0.5*q2vR)
            cwR = 0.5 / c * gm1*(HwR - 0.5*q2wR)
            cpR = 0.5 / c * gm1*(HpR - 0.5*q2pR)
!
            ubar = xnorm*u + ynorm*v + znorm*w
!
            ubarrL = xnorm*urL + ynorm*vrL + znorm*wrL
            ubaruL = xnorm*uuL + ynorm*vuL + znorm*wuL
            ubarvL = xnorm*uvL + ynorm*vvL + znorm*wvL
            ubarwL = xnorm*uwL + ynorm*vwL + znorm*wwL
            ubarpL = xnorm*upL + ynorm*vpL + znorm*wpL
!
            ubarrR = xnorm*urR + ynorm*vrR + znorm*wrR
            ubaruR = xnorm*uuR + ynorm*vuR + znorm*wuR
            ubarvR = xnorm*uvR + ynorm*vvR + znorm*wvR
            ubarwR = xnorm*uwR + ynorm*vwR + znorm*wwR
            ubarpR = xnorm*upR + ynorm*vpR + znorm*wpR
!
! Now compute eigenvalues, eigenvectors, and strengths
!
            eig1 = abs(ubar + c)
            eig2 = abs(ubar - c)
            eig3 = abs(ubar)
            flops = flops + 1521.0
!
            if (ubar+c.gt.0) then
              eig1rL = ubarrL + crL
              eig1uL = ubaruL + cuL
              eig1vL = ubarvL + cvL
              eig1wL = ubarwL + cwL
              eig1pL = ubarpL + cpL
              eig1rR = ubarrR + crR
              eig1uR = ubaruR + cuR
              eig1vR = ubarvR + cvR
              eig1wR = ubarwR + cwR
              eig1pR = ubarpR + cpR
              flops = flops + 10.0
            else
              eig1rL = -(ubarrL + crL)
              eig1uL = -(ubaruL + cuL)
              eig1vL = -(ubarvL + cvL)
              eig1wL = -(ubarwL + cwL)
              eig1pL = -(ubarpL + cpL)
              eig1rR = -(ubarrR + crR)
              eig1uR = -(ubaruR + cuR)
              eig1vR = -(ubarvR + cvR)
              eig1wR = -(ubarwR + cwR)
              eig1pR = -(ubarpR + cpR)
              flops = flops + 20.0
            endif
!
            if (ubar-c.gt.0) then
              eig2rL = ubarrL - crL
              eig2uL = ubaruL - cuL
              eig2vL = ubarvL - cvL
              eig2wL = ubarwL - cwL
              eig2pL = ubarpL - cpL
              eig2rR = ubarrR - crR
              eig2uR = ubaruR - cuR
              eig2vR = ubarvR - cvR
              eig2wR = ubarwR - cwR
              eig2pR = ubarpR - cpR
              flops = flops + 10.0
            else
              eig2rL = -(ubarrL - crL)
              eig2uL = -(ubaruL - cuL)
              eig2vL = -(ubarvL - cvL)
              eig2wL = -(ubarwL - cwL)
              eig2pL = -(ubarpL - cpL)
              eig2rR = -(ubarrR - crR)
              eig2uR = -(ubaruR - cuR)
              eig2vR = -(ubarvR - cvR)
              eig2wR = -(ubarwR - cwR)
              eig2pR = -(ubarpR - cpR)
              flops = flops + 20.0
            endif
!
            if (ubar.gt.0) then
              eig3rL = ubarrL
              eig3uL = ubaruL
              eig3vL = ubarvL
              eig3wL = ubarwL
              eig3pL = ubarpL
              eig3rR = ubarrR
              eig3uR = ubaruR
              eig3vR = ubarvR
              eig3wR = ubarwR
              eig3pR = ubarpR
            else
              eig3rL = -(ubarrL)
              eig3uL = -(ubaruL)
              eig3vL = -(ubarvL)
              eig3wL = -(ubarwL)
              eig3pL = -(ubarpL)
              eig3rR = -(ubarrR)
              eig3uR = -(ubaruR)
              eig3vR = -(ubarvR)
              eig3wR = -(ubarwR)
              eig3pR = -(ubarpR)
              flops = flops + 10.0
            endif
!
            drho   = rhor - rhol
!
            drhorL = rhorrL - rholrL
            drhouL = rhoruL - rholuL
            drhovL = rhorvL - rholvL
            drhowL = rhorwL - rholwL
            drhopL = rhorpL - rholpL
!
            drhorR = rhorrR - rholrR
            drhouR = rhoruR - rholuR
            drhovR = rhorvR - rholvR
            drhowR = rhorwR - rholwR
            drhopR = rhorpR - rholpR
!
            dpress = pressr - pressl
!
            dpressrL = pressrrL - presslrL
            dpressuL = pressruL - pressluL
            dpressvL = pressrvL - presslvL
            dpresswL = pressrwL - presslwL
            dpresspL = pressrpL - presslpL
!
            dpressrR = pressrrR - presslrR
            dpressuR = pressruR - pressluR
            dpressvR = pressrvR - presslvR
            dpresswR = pressrwR - presslwR
            dpresspR = pressrpR - presslpR
!
            du     = ur - ul
!
            durL = urrL - ulrL
            duuL = uruL - uluL
            duvL = urvL - ulvL
            duwL = urwL - ulwL
            dupL = urpL - ulpL
!
            durR = urrR - ulrR
            duuR = uruR - uluR
            duvR = urvR - ulvR
            duwR = urwR - ulwR
            dupR = urpR - ulpR
!
            dv     = vr - vl
!
            dvrL = vrrL - vlrL
            dvuL = vruL - vluL
            dvvL = vrvL - vlvL
            dvwL = vrwL - vlwL
            dvpL = vrpL - vlpL
!
            dvrR = vrrR - vlrR
            dvuR = vruR - vluR
            dvvR = vrvR - vlvR
            dvwR = vrwR - vlwR
            dvpR = vrpR - vlpR
!
            dw     = wr - wl
!
            dwrL = wrrL - wlrL
            dwuL = wruL - wluL
            dwvL = wrvL - wlvL
            dwwL = wrwL - wlwL
            dwpL = wrpL - wlpL
!
            dwrR = wrrR - wlrR
            dwuR = wruR - wluR
            dwvR = wrvR - wlvR
            dwwR = wrwR - wlwR
            dwpR = wrpR - wlpR
!
            dubar  = ubarr - ubarl
!
            dubarrL = ubarrrL - ubarlrL
            dubaruL = ubarruL - ubarluL
            dubarvL = ubarrvL - ubarlvL
            dubarwL = ubarrwL - ubarlwL
            dubarpL = ubarrpL - ubarlpL
!
            dubarrR = ubarrrR - ubarlrR
            dubaruR = ubarruR - ubarluR
            dubarvR = ubarrvR - ubarlvR
            dubarwR = ubarrwR - ubarlwR
            dubarpR = ubarrpR - ubarlpR
!
            c2 = c*c
!
            c2rL = 2. * c * crL
            c2uL = 2. * c * cuL
            c2vL = 2. * c * cvL
            c2wL = 2. * c * cwL
            c2pL = 2. * c * cpL
!
            c2rR = 2. * c * crR
            c2uR = 2. * c * cuR
            c2vR = 2. * c * cvR
            c2wR = 2. * c * cwR
            c2pR = 2. * c * cpR
!
! jumps have units of density
!
            dv1 = 0.5*(dpress + rho*c*dubar)/c2
!
            dv1rL = 0.5*(c2*(dpressrL + rho*(c*dubarrL + dubar*crL)
     &              + c*dubar*rhorL) - (dpress + rho*c*dubar)*c2rL)
     &              / c2 / c2
            dv1uL = 0.5*(c2*(dpressuL + rho*(c*dubaruL + dubar*cuL)
     &              + c*dubar*rhouL) - (dpress + rho*c*dubar)*c2uL)
     &              / c2 / c2
            dv1vL = 0.5*(c2*(dpressvL + rho*(c*dubarvL + dubar*cvL)
     &              + c*dubar*rhovL) - (dpress + rho*c*dubar)*c2vL)
     &              / c2 / c2
            dv1wL = 0.5*(c2*(dpresswL + rho*(c*dubarwL + dubar*cwL)
     &              + c*dubar*rhowL) - (dpress + rho*c*dubar)*c2wL)
     &              / c2 / c2
            dv1pL = 0.5*(c2*(dpresspL + rho*(c*dubarpL + dubar*cpL)
     &              + c*dubar*rhopL) - (dpress + rho*c*dubar)*c2pL)
     &              / c2 / c2
!
            dv1rR = 0.5*(c2*(dpressrR + rho*(c*dubarrR + dubar*crR)
     &              + c*dubar*rhorR) - (dpress + rho*c*dubar)*c2rR)
     &              / c2 / c2
            dv1uR = 0.5*(c2*(dpressuR + rho*(c*dubaruR + dubar*cuR)
     &              + c*dubar*rhouR) - (dpress + rho*c*dubar)*c2uR)
     &              / c2 / c2
            dv1vR = 0.5*(c2*(dpressvR + rho*(c*dubarvR + dubar*cvR)
     &              + c*dubar*rhovR) - (dpress + rho*c*dubar)*c2vR)
     &              / c2 / c2
            dv1wR = 0.5*(c2*(dpresswR + rho*(c*dubarwR + dubar*cwR)
     &              + c*dubar*rhowR) - (dpress + rho*c*dubar)*c2wR)
     &              / c2 / c2
            dv1pR = 0.5*(c2*(dpresspR + rho*(c*dubarpR + dubar*cpR)
     &              + c*dubar*rhopR) - (dpress + rho*c*dubar)*c2pR)
     &              / c2 / c2
!
!
            dv2 = 0.5*(dpress - rho*c*dubar)/c2
!
            dv2rL = 0.5*(c2*(dpressrL - rho*(c*dubarrL + dubar*crL)
     &              - c*dubar*rhorL) - (dpress - rho*c*dubar)*c2rL)
     &              / c2 / c2
            dv2uL = 0.5*(c2*(dpressuL - rho*(c*dubaruL + dubar*cuL)
     &              - c*dubar*rhouL) - (dpress - rho*c*dubar)*c2uL)
     &              / c2 / c2
            dv2vL = 0.5*(c2*(dpressvL - rho*(c*dubarvL + dubar*cvL)
     &              - c*dubar*rhovL) - (dpress - rho*c*dubar)*c2vL)
     &              / c2 / c2
            dv2wL = 0.5*(c2*(dpresswL - rho*(c*dubarwL + dubar*cwL)
     &              - c*dubar*rhowL) - (dpress - rho*c*dubar)*c2wL)
     &              / c2 / c2
            dv2pL = 0.5*(c2*(dpresspL - rho*(c*dubarpL + dubar*cpL)
     &              - c*dubar*rhopL) - (dpress - rho*c*dubar)*c2pL)
     &              / c2 / c2
!
            dv2rR = 0.5*(c2*(dpressrR - rho*(c*dubarrR + dubar*crR)
     &              - c*dubar*rhorR) - (dpress - rho*c*dubar)*c2rR)
     &              / c2 / c2
            dv2uR = 0.5*(c2*(dpressuR - rho*(c*dubaruR + dubar*cuR)
     &              - c*dubar*rhouR) - (dpress - rho*c*dubar)*c2uR)
     &              / c2 / c2
            dv2vR = 0.5*(c2*(dpressvR - rho*(c*dubarvR + dubar*cvR)
     &              - c*dubar*rhovR) - (dpress - rho*c*dubar)*c2vR)
     &              / c2 / c2
            dv2wR = 0.5*(c2*(dpresswR - rho*(c*dubarwR + dubar*cwR)
     &              - c*dubar*rhowR) - (dpress - rho*c*dubar)*c2wR)
     &              / c2 / c2
            dv2pR = 0.5*(c2*(dpresspR - rho*(c*dubarpR + dubar*cpR)
     &              - c*dubar*rhopR) - (dpress - rho*c*dubar)*c2pR)
     &              / c2 / c2
!
            dv3 = rho
!
            dv3rL = rhorL
            dv3uL = rhouL
            dv3vL = rhovL
            dv3wL = rhowL
            dv3pL = rhopL
!
            dv3rR = rhorR
            dv3uR = rhouR
            dv3vR = rhovR
            dv3wR = rhowR
            dv3pR = rhopR
!
            dv4 = (c*c*drho - dpress)/c2
!
            dv4rL = (c2*((c*(c*drhorL+drho*crL)+c*drho*crL) - dpressrL)
     &              - (c*c*drho - dpress)*c2rL) / c2 / c2
            dv4uL = (c2*((c*(c*drhouL+drho*cuL)+c*drho*cuL) - dpressuL)
     &              - (c*c*drho - dpress)*c2uL) / c2 / c2
            dv4vL = (c2*((c*(c*drhovL+drho*cvL)+c*drho*cvL) - dpressvL)
     &              - (c*c*drho - dpress)*c2vL) / c2 / c2
            dv4wL = (c2*((c*(c*drhowL+drho*cwL)+c*drho*cwL) - dpresswL)
     &              - (c*c*drho - dpress)*c2wL) / c2 / c2
            dv4pL = (c2*((c*(c*drhopL+drho*cpL)+c*drho*cpL) - dpresspL)
     &              - (c*c*drho - dpress)*c2pL) / c2 / c2
!
            dv4rR = (c2*((c*(c*drhorR+drho*crR)+c*drho*crR) - dpressrR)
     &              - (c*c*drho - dpress)*c2rR) / c2 / c2
            dv4uR = (c2*((c*(c*drhouR+drho*cuR)+c*drho*cuR) - dpressuR)
     &              - (c*c*drho - dpress)*c2uR) / c2 / c2
            dv4vR = (c2*((c*(c*drhovR+drho*cvR)+c*drho*cvR) - dpressvR)
     &              - (c*c*drho - dpress)*c2vR) / c2 / c2
            dv4wR = (c2*((c*(c*drhowR+drho*cwR)+c*drho*cwR) - dpresswR)
     &              - (c*c*drho - dpress)*c2wR) / c2 / c2
            dv4pR = (c2*((c*(c*drhopR+drho*cpR)+c*drho*cpR) - dpresspR)
     &              - (c*c*drho - dpress)*c2pR) / c2 / c2
!
            r21 = u + c*xnorm
!
            r21rL = urL + xnorm*crL
            r21uL = uuL + xnorm*cuL
            r21vL = uvL + xnorm*cvL
            r21wL = uwL + xnorm*cwL
            r21pL = upL + xnorm*cpL
!
            r21rR = urR + xnorm*crR
            r21uR = uuR + xnorm*cuR
            r21vR = uvR + xnorm*cvR
            r21wR = uwR + xnorm*cwR
            r21pR = upR + xnorm*cpR
!
            r31 = v + c*ynorm
!
            r31rL = vrL + ynorm*crL
            r31uL = vuL + ynorm*cuL
            r31vL = vvL + ynorm*cvL
            r31wL = vwL + ynorm*cwL
            r31pL = vpL + ynorm*cpL
!
            r31rR = vrR + ynorm*crR
            r31uR = vuR + ynorm*cuR
            r31vR = vvR + ynorm*cvR
            r31wR = vwR + ynorm*cwR
            r31pR = vpR + ynorm*cpR
!
            r41 = w + c*znorm
!
            r41rL = wrL + znorm*crL
            r41uL = wuL + znorm*cuL
            r41vL = wvL + znorm*cvL
            r41wL = wwL + znorm*cwL
            r41pL = wpL + znorm*cpL
!
            r41rR = wrR + znorm*crR
            r41uR = wuR + znorm*cuR
            r41vR = wvR + znorm*cvR
            r41wR = wwR + znorm*cwR
            r41pR = wpR + znorm*cpR
!
            r51 = H + c*ubar
!
            r51rL = HrL + c*ubarrL + ubar*crL
            r51uL = HuL + c*ubaruL + ubar*cuL
            r51vL = HvL + c*ubarvL + ubar*cvL
            r51wL = HwL + c*ubarwL + ubar*cwL
            r51pL = HpL + c*ubarpL + ubar*cpL
!
            r51rR = HrR + c*ubarrR + ubar*crR
            r51uR = HuR + c*ubaruR + ubar*cuR
            r51vR = HvR + c*ubarvR + ubar*cvR
            r51wR = HwR + c*ubarwR + ubar*cwR
            r51pR = HpR + c*ubarpR + ubar*cpR
!
            r22 = u - c*xnorm
!
            r22rL = urL - xnorm*crL
            r22uL = uuL - xnorm*cuL
            r22vL = uvL - xnorm*cvL
            r22wL = uwL - xnorm*cwL
            r22pL = upL - xnorm*cpL
!
            r22rR = urR - xnorm*crR
            r22uR = uuR - xnorm*cuR
            r22vR = uvR - xnorm*cvR
            r22wR = uwR - xnorm*cwR
            r22pR = upR - xnorm*cpR
!
            r32 = v - c*ynorm
!
            r32rL = vrL - ynorm*crL
            r32uL = vuL - ynorm*cuL
            r32vL = vvL - ynorm*cvL
            r32wL = vwL - ynorm*cwL
            r32pL = vpL - ynorm*cpL
!
            r32rR = vrR - ynorm*crR
            r32uR = vuR - ynorm*cuR
            r32vR = vvR - ynorm*cvR
            r32wR = vwR - ynorm*cwR
            r32pR = vpR - ynorm*cpR
!
            r42 = w - c*znorm
!
            r42rL = wrL - znorm*crL
            r42uL = wuL - znorm*cuL
            r42vL = wvL - znorm*cvL
            r42wL = wwL - znorm*cwL
            r42pL = wpL - znorm*cpL
!
            r42rR = wrR - znorm*crR
            r42uR = wuR - znorm*cuR
            r42vR = wvR - znorm*cvR
            r42wR = wwR - znorm*cwR
            r42pR = wpR - znorm*cpR
!
            r52 = H - c*ubar
!
            r52rL = HrL - c*ubarrL - ubar*crL
            r52uL = HuL - c*ubaruL - ubar*cuL
            r52vL = HvL - c*ubarvL - ubar*cvL
            r52wL = HwL - c*ubarwL - ubar*cwL
            r52pL = HpL - c*ubarpL - ubar*cpL
!
            r52rR = HrR - c*ubarrR - ubar*crR
            r52uR = HuR - c*ubaruR - ubar*cuR
            r52vR = HvR - c*ubarvR - ubar*cvR
            r52wR = HwR - c*ubarwR - ubar*cwR
            r52pR = HpR - c*ubarpR - ubar*cpR
!
            r23 = du - dubar*xnorm
!
            r23rL = durL - xnorm*dubarrL
            r23uL = duuL - xnorm*dubaruL
            r23vL = duvL - xnorm*dubarvL
            r23wL = duwL - xnorm*dubarwL
            r23pL = dupL - xnorm*dubarpL
!
            r23rR = durR - xnorm*dubarrR
            r23uR = duuR - xnorm*dubaruR
            r23vR = duvR - xnorm*dubarvR
            r23wR = duwR - xnorm*dubarwR
            r23pR = dupR - xnorm*dubarpR
!
            r33 = dv - dubar*ynorm
!
            r33rL = dvrL - ynorm*dubarrL
            r33uL = dvuL - ynorm*dubaruL
            r33vL = dvvL - ynorm*dubarvL
            r33wL = dvwL - ynorm*dubarwL
            r33pL = dvpL - ynorm*dubarpL
!
            r33rR = dvrR - ynorm*dubarrR
            r33uR = dvuR - ynorm*dubaruR
            r33vR = dvvR - ynorm*dubarvR
            r33wR = dvwR - ynorm*dubarwR
            r33pR = dvpR - ynorm*dubarpR
!
            r43 = dw - dubar*znorm
!
            r43rL = dwrL - znorm*dubarrL
            r43uL = dwuL - znorm*dubaruL
            r43vL = dwvL - znorm*dubarvL
            r43wL = dwwL - znorm*dubarwL
            r43pL = dwpL - znorm*dubarpL
!
            r43rR = dwrR - znorm*dubarrR
            r43uR = dwuR - znorm*dubaruR
            r43vR = dwvR - znorm*dubarvR
            r43wR = dwwR - znorm*dubarwR
            r43pR = dwpR - znorm*dubarpR
!
            r53 = u*du + v*dv + w*dw - ubar*dubar
!
            r53rL = u*durL+du*urL + v*dvrL+dv*vrL + w*dwrL+dw*wrL
     &              - ubar*dubarrL - dubar*ubarrL
            r53uL = u*duuL+du*uuL + v*dvuL+dv*vuL + w*dwuL+dw*wuL
     &              - ubar*dubaruL - dubar*ubaruL
            r53vL = u*duvL+du*uvL + v*dvvL+dv*vvL + w*dwvL+dw*wvL
     &              - ubar*dubarvL - dubar*ubarvL
            r53wL = u*duwL+du*uwL + v*dvwL+dv*vwL + w*dwwL+dw*wwL
     &              - ubar*dubarwL - dubar*ubarwL
            r53pL = u*dupL+du*upL + v*dvpL+dv*vpL + w*dwpL+dw*wpL
     &              - ubar*dubarpL - dubar*ubarpL
!
            r53rR = u*durR+du*urR + v*dvrR+dv*vrR + w*dwrR+dw*wrR
     &              - ubar*dubarrR - dubar*ubarrR
            r53uR = u*duuR+du*uuR + v*dvuR+dv*vuR + w*dwuR+dw*wuR
     &              - ubar*dubaruR - dubar*ubaruR
            r53vR = u*duvR+du*uvR + v*dvvR+dv*vvR + w*dwvR+dw*wvR
     &              - ubar*dubarvR - dubar*ubarvR
            r53wR = u*duwR+du*uwR + v*dvwR+dv*vwR + w*dwwR+dw*wwR
     &              - ubar*dubarwR - dubar*ubarwR
            r53pR = u*dupR+du*upR + v*dvpR+dv*vpR + w*dwpR+dw*wpR
     &              - ubar*dubarpR - dubar*ubarpR
!
            r24 = u
!
            r24rL = urL
            r24uL = uuL
            r24vL = uvL
            r24wL = uwL
            r24pL = upL
!
            r24rR = urR
            r24uR = uuR
            r24vR = uvR
            r24wR = uwR
            r24pR = upR
!
            r34 = v
!
            r34rL = vrL
            r34uL = vuL
            r34vL = vvL
            r34wL = vwL
            r34pL = vpL
!
            r34rR = vrR
            r34uR = vuR
            r34vR = vvR
            r34wR = vwR
            r34pR = vpR

            r44 = w
!
            r44rL = wrL
            r44uL = wuL
            r44vL = wvL
            r44wL = wwL
            r44pL = wpL
!
            r44rR = wrR
            r44uR = wuR
            r44vR = wvR
            r44wR = wwR
            r44pR = wpR
!
            r54 = 0.5*q2
!
            r54rL = 0.5*q2rL
            r54uL = 0.5*q2uL
            r54vL = 0.5*q2vL
            r54wL = 0.5*q2wL
            r54pL = 0.5*q2pL
!
            r54rR = 0.5*q2rR
            r54uR = 0.5*q2uR
            r54vR = 0.5*q2vR
            r54wR = 0.5*q2wR
            r54pR = 0.5*q2pR
!
            t1 = eig1*dv1     + eig2*dv2
     &                        + eig3*dv4
!
            t1rL = eig1*dv1rL+dv1*eig1rL + eig2*dv2rL+dv2*eig2rL
     &           + eig3*dv4rL+dv4*eig3rL
            t1uL = eig1*dv1uL+dv1*eig1uL + eig2*dv2uL+dv2*eig2uL
     &           + eig3*dv4uL+dv4*eig3uL
            t1vL = eig1*dv1vL+dv1*eig1vL + eig2*dv2vL+dv2*eig2vL
     &           + eig3*dv4vL+dv4*eig3vL
            t1wL = eig1*dv1wL+dv1*eig1wL + eig2*dv2wL+dv2*eig2wL
     &           + eig3*dv4wL+dv4*eig3wL
            t1pL = eig1*dv1pL+dv1*eig1pL + eig2*dv2pL+dv2*eig2pL
     &           + eig3*dv4pL+dv4*eig3pL
!
            t1rR = eig1*dv1rR+dv1*eig1rR + eig2*dv2rR+dv2*eig2rR
     &           + eig3*dv4rR+dv4*eig3rR
            t1uR = eig1*dv1uR+dv1*eig1uR + eig2*dv2uR+dv2*eig2uR
     &           + eig3*dv4uR+dv4*eig3uR
            t1vR = eig1*dv1vR+dv1*eig1vR + eig2*dv2vR+dv2*eig2vR
     &           + eig3*dv4vR+dv4*eig3vR
            t1wR = eig1*dv1wR+dv1*eig1wR + eig2*dv2wR+dv2*eig2wR
     &           + eig3*dv4wR+dv4*eig3wR
            t1pR = eig1*dv1pR+dv1*eig1pR + eig2*dv2pR+dv2*eig2pR
     &           + eig3*dv4pR+dv4*eig3pR
!
            t2 = eig1*r21*dv1 + eig2*r22*dv2
     &         + eig3*r23*dv3 + eig3*r24*dv4
!
            t2rL = eig1*(r21*dv1rL+dv1*r21rL)+r21*dv1*eig1rL
     &           + eig2*(r22*dv2rL+dv2*r22rL)+r22*dv2*eig2rL
     &           + eig3*(r23*dv3rL+dv3*r23rL)+r23*dv3*eig3rL
     &           + eig3*(r24*dv4rL+dv4*r24rL)+r24*dv4*eig3rL
!
            t2uL = eig1*(r21*dv1uL+dv1*r21uL)+r21*dv1*eig1uL
     &           + eig2*(r22*dv2uL+dv2*r22uL)+r22*dv2*eig2uL
     &           + eig3*(r23*dv3uL+dv3*r23uL)+r23*dv3*eig3uL
     &           + eig3*(r24*dv4uL+dv4*r24uL)+r24*dv4*eig3uL
!
            t2vL = eig1*(r21*dv1vL+dv1*r21vL)+r21*dv1*eig1vL
     &           + eig2*(r22*dv2vL+dv2*r22vL)+r22*dv2*eig2vL
     &           + eig3*(r23*dv3vL+dv3*r23vL)+r23*dv3*eig3vL
     &           + eig3*(r24*dv4vL+dv4*r24vL)+r24*dv4*eig3vL
!
            t2wL = eig1*(r21*dv1wL+dv1*r21wL)+r21*dv1*eig1wL
     &           + eig2*(r22*dv2wL+dv2*r22wL)+r22*dv2*eig2wL
     &           + eig3*(r23*dv3wL+dv3*r23wL)+r23*dv3*eig3wL
     &           + eig3*(r24*dv4wL+dv4*r24wL)+r24*dv4*eig3wL
!
            t2pL = eig1*(r21*dv1pL+dv1*r21pL)+r21*dv1*eig1pL
     &           + eig2*(r22*dv2pL+dv2*r22pL)+r22*dv2*eig2pL
     &           + eig3*(r23*dv3pL+dv3*r23pL)+r23*dv3*eig3pL
     &           + eig3*(r24*dv4pL+dv4*r24pL)+r24*dv4*eig3pL
!
!
            t2rR = eig1*(r21*dv1rR+dv1*r21rR)+r21*dv1*eig1rR
     &           + eig2*(r22*dv2rR+dv2*r22rR)+r22*dv2*eig2rR
     &           + eig3*(r23*dv3rR+dv3*r23rR)+r23*dv3*eig3rR
     &           + eig3*(r24*dv4rR+dv4*r24rR)+r24*dv4*eig3rR
!
            t2uR = eig1*(r21*dv1uR+dv1*r21uR)+r21*dv1*eig1uR
     &           + eig2*(r22*dv2uR+dv2*r22uR)+r22*dv2*eig2uR
     &           + eig3*(r23*dv3uR+dv3*r23uR)+r23*dv3*eig3uR
     &           + eig3*(r24*dv4uR+dv4*r24uR)+r24*dv4*eig3uR
!
            t2vR = eig1*(r21*dv1vR+dv1*r21vR)+r21*dv1*eig1vR
     &           + eig2*(r22*dv2vR+dv2*r22vR)+r22*dv2*eig2vR
     &           + eig3*(r23*dv3vR+dv3*r23vR)+r23*dv3*eig3vR
     &           + eig3*(r24*dv4vR+dv4*r24vR)+r24*dv4*eig3vR
!
            t2wR = eig1*(r21*dv1wR+dv1*r21wR)+r21*dv1*eig1wR
     &           + eig2*(r22*dv2wR+dv2*r22wR)+r22*dv2*eig2wR
     &           + eig3*(r23*dv3wR+dv3*r23wR)+r23*dv3*eig3wR
     &           + eig3*(r24*dv4wR+dv4*r24wR)+r24*dv4*eig3wR
!
            t2pR = eig1*(r21*dv1pR+dv1*r21pR)+r21*dv1*eig1pR
     &           + eig2*(r22*dv2pR+dv2*r22pR)+r22*dv2*eig2pR
     &           + eig3*(r23*dv3pR+dv3*r23pR)+r23*dv3*eig3pR
     &           + eig3*(r24*dv4pR+dv4*r24pR)+r24*dv4*eig3pR
!
!
            t3 = eig1*r31*dv1 + eig2*r32*dv2
     &         + eig3*r33*dv3 + eig3*r34*dv4
!
            t3rL = eig1*(r31*dv1rL+dv1*r31rL)+r31*dv1*eig1rL
     &           + eig2*(r32*dv2rL+dv2*r32rL)+r32*dv2*eig2rL
     &           + eig3*(r33*dv3rL+dv3*r33rL)+r33*dv3*eig3rL
     &           + eig3*(r34*dv4rL+dv4*r34rL)+r34*dv4*eig3rL
!
            t3uL = eig1*(r31*dv1uL+dv1*r31uL)+r31*dv1*eig1uL
     &           + eig2*(r32*dv2uL+dv2*r32uL)+r32*dv2*eig2uL
     &           + eig3*(r33*dv3uL+dv3*r33uL)+r33*dv3*eig3uL
     &           + eig3*(r34*dv4uL+dv4*r34uL)+r34*dv4*eig3uL
!
            t3vL = eig1*(r31*dv1vL+dv1*r31vL)+r31*dv1*eig1vL
     &           + eig2*(r32*dv2vL+dv2*r32vL)+r32*dv2*eig2vL
     &           + eig3*(r33*dv3vL+dv3*r33vL)+r33*dv3*eig3vL
     &           + eig3*(r34*dv4vL+dv4*r34vL)+r34*dv4*eig3vL
!
            t3wL = eig1*(r31*dv1wL+dv1*r31wL)+r31*dv1*eig1wL
     &           + eig2*(r32*dv2wL+dv2*r32wL)+r32*dv2*eig2wL
     &           + eig3*(r33*dv3wL+dv3*r33wL)+r33*dv3*eig3wL
     &           + eig3*(r34*dv4wL+dv4*r34wL)+r34*dv4*eig3wL
!
            t3pL = eig1*(r31*dv1pL+dv1*r31pL)+r31*dv1*eig1pL
     &           + eig2*(r32*dv2pL+dv2*r32pL)+r32*dv2*eig2pL
     &           + eig3*(r33*dv3pL+dv3*r33pL)+r33*dv3*eig3pL
     &           + eig3*(r34*dv4pL+dv4*r34pL)+r34*dv4*eig3pL
!
!
            t3rR = eig1*(r31*dv1rR+dv1*r31rR)+r31*dv1*eig1rR
     &           + eig2*(r32*dv2rR+dv2*r32rR)+r32*dv2*eig2rR
     &           + eig3*(r33*dv3rR+dv3*r33rR)+r33*dv3*eig3rR
     &           + eig3*(r34*dv4rR+dv4*r34rR)+r34*dv4*eig3rR
!
            t3uR = eig1*(r31*dv1uR+dv1*r31uR)+r31*dv1*eig1uR
     &           + eig2*(r32*dv2uR+dv2*r32uR)+r32*dv2*eig2uR
     &           + eig3*(r33*dv3uR+dv3*r33uR)+r33*dv3*eig3uR
     &           + eig3*(r34*dv4uR+dv4*r34uR)+r34*dv4*eig3uR
!
            t3vR = eig1*(r31*dv1vR+dv1*r31vR)+r31*dv1*eig1vR
     &           + eig2*(r32*dv2vR+dv2*r32vR)+r32*dv2*eig2vR
     &           + eig3*(r33*dv3vR+dv3*r33vR)+r33*dv3*eig3vR
     &           + eig3*(r34*dv4vR+dv4*r34vR)+r34*dv4*eig3vR
!
            t3wR = eig1*(r31*dv1wR+dv1*r31wR)+r31*dv1*eig1wR
     &           + eig2*(r32*dv2wR+dv2*r32wR)+r32*dv2*eig2wR
     &           + eig3*(r33*dv3wR+dv3*r33wR)+r33*dv3*eig3wR
     &           + eig3*(r34*dv4wR+dv4*r34wR)+r34*dv4*eig3wR
!
            t3pR = eig1*(r31*dv1pR+dv1*r31pR)+r31*dv1*eig1pR
     &           + eig2*(r32*dv2pR+dv2*r32pR)+r32*dv2*eig2pR
     &           + eig3*(r33*dv3pR+dv3*r33pR)+r33*dv3*eig3pR
     &           + eig3*(r34*dv4pR+dv4*r34pR)+r34*dv4*eig3pR
!
!
            t4 = eig1*r41*dv1 + eig2*r42*dv2
     &         + eig3*r43*dv3 + eig3*r44*dv4
!
            t4rL = eig1*(r41*dv1rL+dv1*r41rL)+r41*dv1*eig1rL
     &           + eig2*(r42*dv2rL+dv2*r42rL)+r42*dv2*eig2rL
     &           + eig3*(r43*dv3rL+dv3*r43rL)+r43*dv3*eig3rL
     &           + eig3*(r44*dv4rL+dv4*r44rL)+r44*dv4*eig3rL
!
            t4uL = eig1*(r41*dv1uL+dv1*r41uL)+r41*dv1*eig1uL
     &           + eig2*(r42*dv2uL+dv2*r42uL)+r42*dv2*eig2uL
     &           + eig3*(r43*dv3uL+dv3*r43uL)+r43*dv3*eig3uL
     &           + eig3*(r44*dv4uL+dv4*r44uL)+r44*dv4*eig3uL
!
            t4vL = eig1*(r41*dv1vL+dv1*r41vL)+r41*dv1*eig1vL
     &           + eig2*(r42*dv2vL+dv2*r42vL)+r42*dv2*eig2vL
     &           + eig3*(r43*dv3vL+dv3*r43vL)+r43*dv3*eig3vL
     &           + eig3*(r44*dv4vL+dv4*r44vL)+r44*dv4*eig3vL
!
            t4wL = eig1*(r41*dv1wL+dv1*r41wL)+r41*dv1*eig1wL
     &           + eig2*(r42*dv2wL+dv2*r42wL)+r42*dv2*eig2wL
     &           + eig3*(r43*dv3wL+dv3*r43wL)+r43*dv3*eig3wL
     &           + eig3*(r44*dv4wL+dv4*r44wL)+r44*dv4*eig3wL
!
            t4pL = eig1*(r41*dv1pL+dv1*r41pL)+r41*dv1*eig1pL
     &           + eig2*(r42*dv2pL+dv2*r42pL)+r42*dv2*eig2pL
     &           + eig3*(r43*dv3pL+dv3*r43pL)+r43*dv3*eig3pL
     &           + eig3*(r44*dv4pL+dv4*r44pL)+r44*dv4*eig3pL
!
!
            t4rR = eig1*(r41*dv1rR+dv1*r41rR)+r41*dv1*eig1rR
     &           + eig2*(r42*dv2rR+dv2*r42rR)+r42*dv2*eig2rR
     &           + eig3*(r43*dv3rR+dv3*r43rR)+r43*dv3*eig3rR
     &           + eig3*(r44*dv4rR+dv4*r44rR)+r44*dv4*eig3rR
!
            t4uR = eig1*(r41*dv1uR+dv1*r41uR)+r41*dv1*eig1uR
     &           + eig2*(r42*dv2uR+dv2*r42uR)+r42*dv2*eig2uR
     &           + eig3*(r43*dv3uR+dv3*r43uR)+r43*dv3*eig3uR
     &           + eig3*(r44*dv4uR+dv4*r44uR)+r44*dv4*eig3uR
!
            t4vR = eig1*(r41*dv1vR+dv1*r41vR)+r41*dv1*eig1vR
     &           + eig2*(r42*dv2vR+dv2*r42vR)+r42*dv2*eig2vR
     &           + eig3*(r43*dv3vR+dv3*r43vR)+r43*dv3*eig3vR
     &           + eig3*(r44*dv4vR+dv4*r44vR)+r44*dv4*eig3vR
!
            t4wR = eig1*(r41*dv1wR+dv1*r41wR)+r41*dv1*eig1wR
     &           + eig2*(r42*dv2wR+dv2*r42wR)+r42*dv2*eig2wR
     &           + eig3*(r43*dv3wR+dv3*r43wR)+r43*dv3*eig3wR
     &           + eig3*(r44*dv4wR+dv4*r44wR)+r44*dv4*eig3wR
!
            t4pR = eig1*(r41*dv1pR+dv1*r41pR)+r41*dv1*eig1pR
     &           + eig2*(r42*dv2pR+dv2*r42pR)+r42*dv2*eig2pR
     &           + eig3*(r43*dv3pR+dv3*r43pR)+r43*dv3*eig3pR
     &           + eig3*(r44*dv4pR+dv4*r44pR)+r44*dv4*eig3pR
!
!
            t5 = eig1*r51*dv1 + eig2*r52*dv2
     &         + eig3*r53*dv3 + eig3*r54*dv4
!
            t5rL = eig1*(r51*dv1rL+dv1*r51rL)+r51*dv1*eig1rL
     &           + eig2*(r52*dv2rL+dv2*r52rL)+r52*dv2*eig2rL
     &           + eig3*(r53*dv3rL+dv3*r53rL)+r53*dv3*eig3rL
     &           + eig3*(r54*dv4rL+dv4*r54rL)+r54*dv4*eig3rL
!
            t5uL = eig1*(r51*dv1uL+dv1*r51uL)+r51*dv1*eig1uL
     &           + eig2*(r52*dv2uL+dv2*r52uL)+r52*dv2*eig2uL
     &           + eig3*(r53*dv3uL+dv3*r53uL)+r53*dv3*eig3uL
     &           + eig3*(r54*dv4uL+dv4*r54uL)+r54*dv4*eig3uL
!
            t5vL = eig1*(r51*dv1vL+dv1*r51vL)+r51*dv1*eig1vL
     &           + eig2*(r52*dv2vL+dv2*r52vL)+r52*dv2*eig2vL
     &           + eig3*(r53*dv3vL+dv3*r53vL)+r53*dv3*eig3vL
     &           + eig3*(r54*dv4vL+dv4*r54vL)+r54*dv4*eig3vL
!
            t5wL = eig1*(r51*dv1wL+dv1*r51wL)+r51*dv1*eig1wL
     &           + eig2*(r52*dv2wL+dv2*r52wL)+r52*dv2*eig2wL
     &           + eig3*(r53*dv3wL+dv3*r53wL)+r53*dv3*eig3wL
     &           + eig3*(r54*dv4wL+dv4*r54wL)+r54*dv4*eig3wL
!
            t5pL = eig1*(r51*dv1pL+dv1*r51pL)+r51*dv1*eig1pL
     &           + eig2*(r52*dv2pL+dv2*r52pL)+r52*dv2*eig2pL
     &           + eig3*(r53*dv3pL+dv3*r53pL)+r53*dv3*eig3pL
     &           + eig3*(r54*dv4pL+dv4*r54pL)+r54*dv4*eig3pL
!
!
            t5rR = eig1*(r51*dv1rR+dv1*r51rR)+r51*dv1*eig1rR
     &           + eig2*(r52*dv2rR+dv2*r52rR)+r52*dv2*eig2rR
     &           + eig3*(r53*dv3rR+dv3*r53rR)+r53*dv3*eig3rR
     &           + eig3*(r54*dv4rR+dv4*r54rR)+r54*dv4*eig3rR
!
            t5uR = eig1*(r51*dv1uR+dv1*r51uR)+r51*dv1*eig1uR
     &           + eig2*(r52*dv2uR+dv2*r52uR)+r52*dv2*eig2uR
     &           + eig3*(r53*dv3uR+dv3*r53uR)+r53*dv3*eig3uR
     &           + eig3*(r54*dv4uR+dv4*r54uR)+r54*dv4*eig3uR
!
            t5vR = eig1*(r51*dv1vR+dv1*r51vR)+r51*dv1*eig1vR
     &           + eig2*(r52*dv2vR+dv2*r52vR)+r52*dv2*eig2vR
     &           + eig3*(r53*dv3vR+dv3*r53vR)+r53*dv3*eig3vR
     &           + eig3*(r54*dv4vR+dv4*r54vR)+r54*dv4*eig3vR
!
            t5wR = eig1*(r51*dv1wR+dv1*r51wR)+r51*dv1*eig1wR
     &           + eig2*(r52*dv2wR+dv2*r52wR)+r52*dv2*eig2wR
     &           + eig3*(r53*dv3wR+dv3*r53wR)+r53*dv3*eig3wR
     &           + eig3*(r54*dv4wR+dv4*r54wR)+r54*dv4*eig3wR
!
            t5pR = eig1*(r51*dv1pR+dv1*r51pR)+r51*dv1*eig1pR
     &           + eig2*(r52*dv2pR+dv2*r52pR)+r52*dv2*eig2pR
     &           + eig3*(r53*dv3pR+dv3*r53pR)+r53*dv3*eig3pR
     &           + eig3*(r54*dv4pR+dv4*r54pR)+r54*dv4*eig3pR
!
! Compute flux using variables from left side of face
!
            fluxp1 = area*rhol*ubarl
!
            fluxp1rL = area*(rhol*ubarlrL + ubarl*rholrL)
            fluxp1uL = area*(rhol*ubarluL + ubarl*rholuL)
            fluxp1vL = area*(rhol*ubarlvL + ubarl*rholvL)
            fluxp1wL = area*(rhol*ubarlwL + ubarl*rholwL)
            fluxp1pL = area*(rhol*ubarlpL + ubarl*rholpL)
!
            fluxp1rR = area*(rhol*ubarlrR + ubarl*rholrR)
            fluxp1uR = area*(rhol*ubarluR + ubarl*rholuR)
            fluxp1vR = area*(rhol*ubarlvR + ubarl*rholvR)
            fluxp1wR = area*(rhol*ubarlwR + ubarl*rholwR)
            fluxp1pR = area*(rhol*ubarlpR + ubarl*rholpR)
!
            fluxp2 = area*(rhol*ul*ubarl + xnorm*pressl)
!
            fluxp2rL = area*(rhol*(ul*ubarlrL+ubarl*ulrL) +
     &                 ul*ubarl*rholrL + xnorm*presslrL)
            fluxp2uL = area*(rhol*(ul*ubarluL+ubarl*uluL) +
     &                 ul*ubarl*rholuL + xnorm*pressluL)
            fluxp2vL = area*(rhol*(ul*ubarlvL+ubarl*ulvL) +
     &                 ul*ubarl*rholvL + xnorm*presslvL)
            fluxp2wL = area*(rhol*(ul*ubarlwL+ubarl*ulwL) +
     &                 ul*ubarl*rholwL + xnorm*presslwL)
            fluxp2pL = area*(rhol*(ul*ubarlpL+ubarl*ulpL) +
     &                 ul*ubarl*rholpL + xnorm*presslpL)
!
            fluxp2rR = area*(rhol*(ul*ubarlrR+ubarl*ulrR) +
     &                 ul*ubarl*rholrR + xnorm*presslrR)
            fluxp2uR = area*(rhol*(ul*ubarluR+ubarl*uluR) +
     &                 ul*ubarl*rholuR + xnorm*pressluR)
            fluxp2vR = area*(rhol*(ul*ubarlvR+ubarl*ulvR) +
     &                 ul*ubarl*rholvR + xnorm*presslvR)
            fluxp2wR = area*(rhol*(ul*ubarlwR+ubarl*ulwR) +
     &                 ul*ubarl*rholwR + xnorm*presslwR)
            fluxp2pR = area*(rhol*(ul*ubarlpR+ubarl*ulpR) +
     &                 ul*ubarl*rholpR + xnorm*presslpR)
!
!
            fluxp3 = area*(rhol*vl*ubarl + ynorm*pressl)
!
            fluxp3rL = area*(rhol*(vl*ubarlrL+ubarl*vlrL) +
     &                 vl*ubarl*rholrL + ynorm*presslrL)
            fluxp3uL = area*(rhol*(vl*ubarluL+ubarl*vluL) +
     &                 vl*ubarl*rholuL + ynorm*pressluL)
            fluxp3vL = area*(rhol*(vl*ubarlvL+ubarl*vlvL) +
     &                 vl*ubarl*rholvL + ynorm*presslvL)
            fluxp3wL = area*(rhol*(vl*ubarlwL+ubarl*vlwL) +
     &                 vl*ubarl*rholwL + ynorm*presslwL)
            fluxp3pL = area*(rhol*(vl*ubarlpL+ubarl*vlpL) +
     &                 vl*ubarl*rholpL + ynorm*presslpL)
!
            fluxp3rR = area*(rhol*(vl*ubarlrR+ubarl*vlrR) +
     &                 vl*ubarl*rholrR + ynorm*presslrR)
            fluxp3uR = area*(rhol*(vl*ubarluR+ubarl*vluR) +
     &                 vl*ubarl*rholuR + ynorm*pressluR)
            fluxp3vR = area*(rhol*(vl*ubarlvR+ubarl*vlvR) +
     &                 vl*ubarl*rholvR + ynorm*presslvR)
            fluxp3wR = area*(rhol*(vl*ubarlwR+ubarl*vlwR) +
     &                 vl*ubarl*rholwR + ynorm*presslwR)
            fluxp3pR = area*(rhol*(vl*ubarlpR+ubarl*vlpR) +
     &                 vl*ubarl*rholpR + ynorm*presslpR)
!
!
            fluxp4 = area*(rhol*wl*ubarl + znorm*pressl)
!
            fluxp4rL = area*(rhol*(wl*ubarlrL+ubarl*wlrL) +
     &                 wl*ubarl*rholrL + znorm*presslrL)
            fluxp4uL = area*(rhol*(wl*ubarluL+ubarl*wluL) +
     &                 wl*ubarl*rholuL + znorm*pressluL)
            fluxp4vL = area*(rhol*(wl*ubarlvL+ubarl*wlvL) +
     &                 wl*ubarl*rholvL + znorm*presslvL)
            fluxp4wL = area*(rhol*(wl*ubarlwL+ubarl*wlwL) +
     &                 wl*ubarl*rholwL + znorm*presslwL)
            fluxp4pL = area*(rhol*(wl*ubarlpL+ubarl*wlpL) +
     &                 wl*ubarl*rholpL + znorm*presslpL)
!
            fluxp4rR = area*(rhol*(wl*ubarlrR+ubarl*wlrR) +
     &                 wl*ubarl*rholrR + znorm*presslrR)
            fluxp4uR = area*(rhol*(wl*ubarluR+ubarl*wluR) +
     &                 wl*ubarl*rholuR + znorm*pressluR)
            fluxp4vR = area*(rhol*(wl*ubarlvR+ubarl*wlvR) +
     &                 wl*ubarl*rholvR + znorm*presslvR)
            fluxp4wR = area*(rhol*(wl*ubarlwR+ubarl*wlwR) +
     &                 wl*ubarl*rholwR + znorm*presslwR)
            fluxp4pR = area*(rhol*(wl*ubarlpR+ubarl*wlpR) +
     &                 wl*ubarl*rholpR + znorm*presslpR)
!
!
            fluxp5 = area*(enrgyl + pressl)*ubarl
!
            fluxp5rL = area*((enrgyl + pressl)*ubarlrL +
     &                 ubarl*(enrgylrL + presslrL))
            fluxp5uL = area*((enrgyl + pressl)*ubarluL +
     &                 ubarl*(enrgyluL + pressluL))
            fluxp5vL = area*((enrgyl + pressl)*ubarlvL +
     &                 ubarl*(enrgylvL + presslvL))
            fluxp5wL = area*((enrgyl + pressl)*ubarlwL +
     &                 ubarl*(enrgylwL + presslwL))
            fluxp5pL = area*((enrgyl + pressl)*ubarlpL +
     &                 ubarl*(enrgylpL + presslpL))
!
            fluxp5rR = area*((enrgyl + pressl)*ubarlrR +
     &                 ubarl*(enrgylrR + presslrR))
            fluxp5uR = area*((enrgyl + pressl)*ubarluR +
     &                 ubarl*(enrgyluR + pressluR))
            fluxp5vR = area*((enrgyl + pressl)*ubarlvR +
     &                 ubarl*(enrgylvR + presslvR))
            fluxp5wR = area*((enrgyl + pressl)*ubarlwR +
     &                 ubarl*(enrgylwR + presslwR))
            fluxp5pR = area*((enrgyl + pressl)*ubarlpR +
     &                 ubarl*(enrgylpR + presslpR))
!
! Now the right side
!
            fluxm1 = area*rhor*ubarr
!
            fluxm1rL = area*(rhor*ubarrrL + ubarr*rhorrL)
            fluxm1uL = area*(rhor*ubarruL + ubarr*rhoruL)
            fluxm1vL = area*(rhor*ubarrvL + ubarr*rhorvL)
            fluxm1wL = area*(rhor*ubarrwL + ubarr*rhorwL)
            fluxm1pL = area*(rhor*ubarrpL + ubarr*rhorpL)
!
            fluxm1rR = area*(rhor*ubarrrR + ubarr*rhorrR)
            fluxm1uR = area*(rhor*ubarruR + ubarr*rhoruR)
            fluxm1vR = area*(rhor*ubarrvR + ubarr*rhorvR)
            fluxm1wR = area*(rhor*ubarrwR + ubarr*rhorwR)
            fluxm1pR = area*(rhor*ubarrpR + ubarr*rhorpR)
!
!
            fluxm2 = area*(rhor*ur*ubarr + xnorm*pressr)
!
            fluxm2rL = area*(rhor*(ur*ubarrrL+ubarr*urrL) +
     &                 ur*ubarr*rhorrL + xnorm*pressrrL)
            fluxm2uL = area*(rhor*(ur*ubarruL+ubarr*uruL) +
     &                 ur*ubarr*rhoruL + xnorm*pressruL)
            fluxm2vL = area*(rhor*(ur*ubarrvL+ubarr*urvL) +
     &                 ur*ubarr*rhorvL + xnorm*pressrvL)
            fluxm2wL = area*(rhor*(ur*ubarrwL+ubarr*urwL) +
     &                 ur*ubarr*rhorwL + xnorm*pressrwL)
            fluxm2pL = area*(rhor*(ur*ubarrpL+ubarr*urpL) +
     &                 ur*ubarr*rhorpL + xnorm*pressrpL)
!
            fluxm2rR = area*(rhor*(ur*ubarrrR+ubarr*urrR) +
     &                 ur*ubarr*rhorrR + xnorm*pressrrR)
            fluxm2uR = area*(rhor*(ur*ubarruR+ubarr*uruR) +
     &                 ur*ubarr*rhoruR + xnorm*pressruR)
            fluxm2vR = area*(rhor*(ur*ubarrvR+ubarr*urvR) +
     &                 ur*ubarr*rhorvR + xnorm*pressrvR)
            fluxm2wR = area*(rhor*(ur*ubarrwR+ubarr*urwR) +
     &                 ur*ubarr*rhorwR + xnorm*pressrwR)
            fluxm2pR = area*(rhor*(ur*ubarrpR+ubarr*urpR) +
     &                 ur*ubarr*rhorpR + xnorm*pressrpR)
!
!
            fluxm3 = area*(rhor*vr*ubarr + ynorm*pressr)
!
            fluxm3rL = area*(rhor*(vr*ubarrrL+ubarr*vrrL) +
     &                 vr*ubarr*rhorrL + ynorm*pressrrL)
            fluxm3uL = area*(rhor*(vr*ubarruL+ubarr*vruL) +
     &                 vr*ubarr*rhoruL + ynorm*pressruL)
            fluxm3vL = area*(rhor*(vr*ubarrvL+ubarr*vrvL) +
     &                 vr*ubarr*rhorvL + ynorm*pressrvL)
            fluxm3wL = area*(rhor*(vr*ubarrwL+ubarr*vrwL) +
     &                 vr*ubarr*rhorwL + ynorm*pressrwL)
            fluxm3pL = area*(rhor*(vr*ubarrpL+ubarr*vrpL) +
     &                 vr*ubarr*rhorpL + ynorm*pressrpL)
!
            fluxm3rR = area*(rhor*(vr*ubarrrR+ubarr*vrrR) +
     &                 vr*ubarr*rhorrR + ynorm*pressrrR)
            fluxm3uR = area*(rhor*(vr*ubarruR+ubarr*vruR) +
     &                 vr*ubarr*rhoruR + ynorm*pressruR)
            fluxm3vR = area*(rhor*(vr*ubarrvR+ubarr*vrvR) +
     &                 vr*ubarr*rhorvR + ynorm*pressrvR)
            fluxm3wR = area*(rhor*(vr*ubarrwR+ubarr*vrwR) +
     &                 vr*ubarr*rhorwR + ynorm*pressrwR)
            fluxm3pR = area*(rhor*(vr*ubarrpR+ubarr*vrpR) +
     &                 vr*ubarr*rhorpR + ynorm*pressrpR)
!
!
            fluxm4 = area*(rhor*wr*ubarr + znorm*pressr)
!
            fluxm4rL = area*(rhor*(wr*ubarrrL+ubarr*wrrL) +
     &                 wr*ubarr*rhorrL + znorm*pressrrL)
            fluxm4uL = area*(rhor*(wr*ubarruL+ubarr*wruL) +
     &                 wr*ubarr*rhoruL + znorm*pressruL)
            fluxm4vL = area*(rhor*(wr*ubarrvL+ubarr*wrvL) +
     &                 wr*ubarr*rhorvL + znorm*pressrvL)
            fluxm4wL = area*(rhor*(wr*ubarrwL+ubarr*wrwL) +
     &                 wr*ubarr*rhorwL + znorm*pressrwL)
            fluxm4pL = area*(rhor*(wr*ubarrpL+ubarr*wrpL) +
     &                 wr*ubarr*rhorpL + znorm*pressrpL)
!
            fluxm4rR = area*(rhor*(wr*ubarrrR+ubarr*wrrR) +
     &                 wr*ubarr*rhorrR + znorm*pressrrR)
            fluxm4uR = area*(rhor*(wr*ubarruR+ubarr*wruR) +
     &                 wr*ubarr*rhoruR + znorm*pressruR)
            fluxm4vR = area*(rhor*(wr*ubarrvR+ubarr*wrvR) +
     &                 wr*ubarr*rhorvR + znorm*pressrvR)
            fluxm4wR = area*(rhor*(wr*ubarrwR+ubarr*wrwR) +
     &                 wr*ubarr*rhorwR + znorm*pressrwR)
            fluxm4pR = area*(rhor*(wr*ubarrpR+ubarr*wrpR) +
     &                 wr*ubarr*rhorpR + znorm*pressrpR)
!
            fluxm5 = area*(enrgyr + pressr)*ubarr
!
            fluxm5rL = area*((enrgyr + pressr)*ubarrrL +
     &                 ubarr*(enrgyrrL + pressrrL))
            fluxm5uL = area*((enrgyr + pressr)*ubarruL +
     &                 ubarr*(enrgyruL + pressruL))
            fluxm5vL = area*((enrgyr + pressr)*ubarrvL +
     &                 ubarr*(enrgyrvL + pressrvL))
            fluxm5wL = area*((enrgyr + pressr)*ubarrwL +
     &                 ubarr*(enrgyrwL + pressrwL))
            fluxm5pL = area*((enrgyr + pressr)*ubarrpL +
     &                 ubarr*(enrgyrpL + pressrpL))
!
            fluxm5rR = area*((enrgyr + pressr)*ubarrrR +
     &                 ubarr*(enrgyrrR + pressrrR))
            fluxm5uR = area*((enrgyr + pressr)*ubarruR +
     &                 ubarr*(enrgyruR + pressruR))
            fluxm5vR = area*((enrgyr + pressr)*ubarrvR +
     &                 ubarr*(enrgyrvR + pressrvR))
            fluxm5wR = area*((enrgyr + pressr)*ubarrwR +
     &                 ubarr*(enrgyrwR + pressrwR))
            fluxm5pR = area*((enrgyr + pressr)*ubarrpR +
     &                 ubarr*(enrgyrpR + pressrpR))
!
          flux1 = 0.5*(fluxp1 + fluxm1 - area*t1)
          flux2 = 0.5*(fluxp2 + fluxm2 - area*t2)
          flux3 = 0.5*(fluxp3 + fluxm3 - area*t3)
          flux4 = 0.5*(fluxp4 + fluxm4 - area*t4)
          flux5 = 0.5*(fluxp5 + fluxm5 - area*t5)
!
          flux1rL = 0.5*(fluxp1rL + fluxm1rL - area*t1rL)
          flux1uL = 0.5*(fluxp1uL + fluxm1uL - area*t1uL)
          flux1vL = 0.5*(fluxp1vL + fluxm1vL - area*t1vL)
          flux1wL = 0.5*(fluxp1wL + fluxm1wL - area*t1wL)
          flux1pL = 0.5*(fluxp1pL + fluxm1pL - area*t1pL)
          flux1rR = 0.5*(fluxp1rR + fluxm1rR - area*t1rR)
          flux1uR = 0.5*(fluxp1uR + fluxm1uR - area*t1uR)
          flux1vR = 0.5*(fluxp1vR + fluxm1vR - area*t1vR)
          flux1wR = 0.5*(fluxp1wR + fluxm1wR - area*t1wR)
          flux1pR = 0.5*(fluxp1pR + fluxm1pR - area*t1pR)
!
          flux2rL = 0.5*(fluxp2rL + fluxm2rL - area*t2rL)
          flux2uL = 0.5*(fluxp2uL + fluxm2uL - area*t2uL)
          flux2vL = 0.5*(fluxp2vL + fluxm2vL - area*t2vL)
          flux2wL = 0.5*(fluxp2wL + fluxm2wL - area*t2wL)
          flux2pL = 0.5*(fluxp2pL + fluxm2pL - area*t2pL)
          flux2rR = 0.5*(fluxp2rR + fluxm2rR - area*t2rR)
          flux2uR = 0.5*(fluxp2uR + fluxm2uR - area*t2uR)
          flux2vR = 0.5*(fluxp2vR + fluxm2vR - area*t2vR)
          flux2wR = 0.5*(fluxp2wR + fluxm2wR - area*t2wR)
          flux2pR = 0.5*(fluxp2pR + fluxm2pR - area*t2pR)
!
          flux3rL = 0.5*(fluxp3rL + fluxm3rL - area*t3rL)
          flux3uL = 0.5*(fluxp3uL + fluxm3uL - area*t3uL)
          flux3vL = 0.5*(fluxp3vL + fluxm3vL - area*t3vL)
          flux3wL = 0.5*(fluxp3wL + fluxm3wL - area*t3wL)
          flux3pL = 0.5*(fluxp3pL + fluxm3pL - area*t3pL)
          flux3rR = 0.5*(fluxp3rR + fluxm3rR - area*t3rR)
          flux3uR = 0.5*(fluxp3uR + fluxm3uR - area*t3uR)
          flux3vR = 0.5*(fluxp3vR + fluxm3vR - area*t3vR)
          flux3wR = 0.5*(fluxp3wR + fluxm3wR - area*t3wR)
          flux3pR = 0.5*(fluxp3pR + fluxm3pR - area*t3pR)
!
          flux4rL = 0.5*(fluxp4rL + fluxm4rL - area*t4rL)
          flux4uL = 0.5*(fluxp4uL + fluxm4uL - area*t4uL)
          flux4vL = 0.5*(fluxp4vL + fluxm4vL - area*t4vL)
          flux4wL = 0.5*(fluxp4wL + fluxm4wL - area*t4wL)
          flux4pL = 0.5*(fluxp4pL + fluxm4pL - area*t4pL)
          flux4rR = 0.5*(fluxp4rR + fluxm4rR - area*t4rR)
          flux4uR = 0.5*(fluxp4uR + fluxm4uR - area*t4uR)
          flux4vR = 0.5*(fluxp4vR + fluxm4vR - area*t4vR)
          flux4wR = 0.5*(fluxp4wR + fluxm4wR - area*t4wR)
          flux4pR = 0.5*(fluxp4pR + fluxm4pR - area*t4pR)
!
          flux5rL = 0.5*(fluxp5rL + fluxm5rL - area*t5rL)
          flux5uL = 0.5*(fluxp5uL + fluxm5uL - area*t5uL)
          flux5vL = 0.5*(fluxp5vL + fluxm5vL - area*t5vL)
          flux5wL = 0.5*(fluxp5wL + fluxm5wL - area*t5wL)
          flux5pL = 0.5*(fluxp5pL + fluxm5pL - area*t5pL)
          flux5rR = 0.5*(fluxp5rR + fluxm5rR - area*t5rR)
          flux5uR = 0.5*(fluxp5uR + fluxm5uR - area*t5uR)
          flux5vR = 0.5*(fluxp5vR + fluxm5vR - area*t5vR)
          flux5wR = 0.5*(fluxp5wR + fluxm5wR - area*t5wR)
          flux5pR = 0.5*(fluxp5pR + fluxm5pR - area*t5pR)
!/*
!  These last 50 equations are the dfp's and dfm's we're
!  looking for, but they're w.r.t. primitive variables.
!
!  We need to convert these last results into conservative
!  variables, using one last chain-rule.
!
!  q = primitive
!  Q = conservative
!
!  First the left transformation
!  */
        q1Q1L = 1.
        q1Q2L = 0.
        q1Q3L = 0.
        q1Q4L = 0.
        q1Q5L = 0.
!
        q2Q1L = - ul / rhol
        q2Q2L = 1. / rhol
        q2Q3L = 0.
        q2Q4L = 0.
        q2Q5L = 0.
!
        q3Q1L = - vl / rhol
        q3Q2L = 0.
        q3Q3L = 1. / rhol
        q3Q4L = 0.
        q3Q5L = 0.
!
        q4Q1L = - wl / rhol
        q4Q2L = 0.
        q4Q3L = 0.
        q4Q4L = 1. / rhol
        q4Q5L = 0.
!
        q5Q1L = gm1 / 2. * (ul*ul + vl*vl + wl*wl)
        q5Q2L = - gm1 * ul
        q5Q3L = - gm1 * vl
        q5Q4L = - gm1 * wl
        q5Q5L = gm1
!
        dfp(1,1) = flux1rL*q1Q1L + flux1uL*q2Q1L + flux1vL*q3Q1L
     &             + flux1wL*q4Q1L + flux1pL*q5Q1L
!
        dfp(1,2) = flux1rL*q1Q2L + flux1uL*q2Q2L + flux1vL*q3Q2L
     &             + flux1wL*q4Q2L + flux1pL*q5Q2L
!
        dfp(1,3) = flux1rL*q1Q3L + flux1uL*q2Q3L + flux1vL*q3Q3L
     &             + flux1wL*q4Q3L + flux1pL*q5Q3L
!
        dfp(1,4) = flux1rL*q1Q4L + flux1uL*q2Q4L + flux1vL*q3Q4L
     &             + flux1wL*q4Q4L + flux1pL*q5Q4L
!
        dfp(1,5) = flux1rL*q1Q5L + flux1uL*q2Q5L + flux1vL*q3Q5L
     &             + flux1wL*q4Q5L + flux1pL*q5Q5L
!
!
        dfp(2,1) = flux2rL*q1Q1L + flux2uL*q2Q1L + flux2vL*q3Q1L
     &             + flux2wL*q4Q1L + flux2pL*q5Q1L
!
        dfp(2,2) = flux2rL*q1Q2L + flux2uL*q2Q2L + flux2vL*q3Q2L
     &             + flux2wL*q4Q2L + flux2pL*q5Q2L
!
        dfp(2,3) = flux2rL*q1Q3L + flux2uL*q2Q3L + flux2vL*q3Q3L
     &             + flux2wL*q4Q3L + flux2pL*q5Q3L
!
        dfp(2,4) = flux2rL*q1Q4L + flux2uL*q2Q4L + flux2vL*q3Q4L
     &             + flux2wL*q4Q4L + flux2pL*q5Q4L
!
        dfp(2,5) = flux2rL*q1Q5L + flux2uL*q2Q5L + flux2vL*q3Q5L
     &             + flux2wL*q4Q5L + flux2pL*q5Q5L
!
!
        dfp(3,1) = flux3rL*q1Q1L + flux3uL*q2Q1L + flux3vL*q3Q1L
     &             + flux3wL*q4Q1L + flux3pL*q5Q1L
!
        dfp(3,2) = flux3rL*q1Q2L + flux3uL*q2Q2L + flux3vL*q3Q2L
     &             + flux3wL*q4Q2L + flux3pL*q5Q2L
!
        dfp(3,3) = flux3rL*q1Q3L + flux3uL*q2Q3L + flux3vL*q3Q3L
     &             + flux3wL*q4Q3L + flux3pL*q5Q3L
!
        dfp(3,4) = flux3rL*q1Q4L + flux3uL*q2Q4L + flux3vL*q3Q4L
     &             + flux3wL*q4Q4L + flux3pL*q5Q4L
!
        dfp(3,5) = flux3rL*q1Q5L + flux3uL*q2Q5L + flux3vL*q3Q5L
     &             + flux3wL*q4Q5L + flux3pL*q5Q5L
!
!
        dfp(4,1) = flux4rL*q1Q1L + flux4uL*q2Q1L + flux4vL*q3Q1L
     &             + flux4wL*q4Q1L + flux4pL*q5Q1L
!
        dfp(4,2) = flux4rL*q1Q2L + flux4uL*q2Q2L + flux4vL*q3Q2L
     &             + flux4wL*q4Q2L + flux4pL*q5Q2L
!
        dfp(4,3) = flux4rL*q1Q3L + flux4uL*q2Q3L + flux4vL*q3Q3L
     &             + flux4wL*q4Q3L + flux4pL*q5Q3L
!
        dfp(4,4) = flux4rL*q1Q4L + flux4uL*q2Q4L + flux4vL*q3Q4L
     &             + flux4wL*q4Q4L + flux4pL*q5Q4L
!
        dfp(4,5) = flux4rL*q1Q5L + flux4uL*q2Q5L + flux4vL*q3Q5L
     &             + flux4wL*q4Q5L + flux4pL*q5Q5L
!
!
        dfp(5,1) = flux5rL*q1Q1L + flux5uL*q2Q1L + flux5vL*q3Q1L
     &             + flux5wL*q4Q1L + flux5pL*q5Q1L
!
        dfp(5,2) = flux5rL*q1Q2L + flux5uL*q2Q2L + flux5vL*q3Q2L
     &             + flux5wL*q4Q2L + flux5pL*q5Q2L
!
        dfp(5,3) = flux5rL*q1Q3L + flux5uL*q2Q3L + flux5vL*q3Q3L
     &             + flux5wL*q4Q3L + flux5pL*q5Q3L
!
        dfp(5,4) = flux5rL*q1Q4L + flux5uL*q2Q4L + flux5vL*q3Q4L
     &             + flux5wL*q4Q4L + flux5pL*q5Q4L
!
        dfp(5,5) = flux5rL*q1Q5L + flux5uL*q2Q5L + flux5vL*q3Q5L
     &             + flux5wL*q4Q5L + flux5pL*q5Q5L
!
!  Now the right transformation
!
        q1Q1R = 1.
        q1Q2R = 0.
        q1Q3R = 0.
        q1Q4R = 0.
        q1Q5R = 0.
!
        q2Q1R = - ur / rhor
        q2Q2R = 1. / rhor
        q2Q3R = 0.
        q2Q4R = 0.
        q2Q5R = 0.
!
        q3Q1R = - vr / rhor
        q3Q2R = 0.
        q3Q3R = 1. / rhor
        q3Q4R = 0.
        q3Q5R = 0.
!
        q4Q1R = - wr / rhor
        q4Q2R = 0.
        q4Q3R = 0.
        q4Q4R = 1. / rhor
        q4Q5R = 0.
!
        q5Q1R = gm1 / 2. * (ur*ur + vr*vr + wr*wr)
        q5Q2R = - gm1 * ur
        q5Q3R = - gm1 * vr
        q5Q4R = - gm1 * wr
        q5Q5R = gm1
!
!
        dfm(1,1) = flux1rR*q1Q1R + flux1uR*q2Q1R + flux1vR*q3Q1R
     &             + flux1wR*q4Q1R + flux1pR*q5Q1R
!
        dfm(1,2) = flux1rR*q1Q2R + flux1uR*q2Q2R + flux1vR*q3Q2R
     &             + flux1wR*q4Q2R + flux1pR*q5Q2R
!
        dfm(1,3) = flux1rR*q1Q3R + flux1uR*q2Q3R + flux1vR*q3Q3R
     &             + flux1wR*q4Q3R + flux1pR*q5Q3R
!
        dfm(1,4) = flux1rR*q1Q4R + flux1uR*q2Q4R + flux1vR*q3Q4R
     &             + flux1wR*q4Q4R + flux1pR*q5Q4R
!
        dfm(1,5) = flux1rR*q1Q5R + flux1uR*q2Q5R + flux1vR*q3Q5R
     &             + flux1wR*q4Q5R + flux1pR*q5Q5R
!
!
        dfm(2,1) = flux2rR*q1Q1R + flux2uR*q2Q1R + flux2vR*q3Q1R
     &             + flux2wR*q4Q1R + flux2pR*q5Q1R
!
        dfm(2,2) = flux2rR*q1Q2R + flux2uR*q2Q2R + flux2vR*q3Q2R
     &             + flux2wR*q4Q2R + flux2pR*q5Q2R
!
        dfm(2,3) = flux2rR*q1Q3R + flux2uR*q2Q3R + flux2vR*q3Q3R
     &             + flux2wR*q4Q3R + flux2pR*q5Q3R
!
        dfm(2,4) = flux2rR*q1Q4R + flux2uR*q2Q4R + flux2vR*q3Q4R
     &             + flux2wR*q4Q4R + flux2pR*q5Q4R
!
        dfm(2,5) = flux2rR*q1Q5R + flux2uR*q2Q5R + flux2vR*q3Q5R
     &             + flux2wR*q4Q5R + flux2pR*q5Q5R
!
!
        dfm(3,1) = flux3rR*q1Q1R + flux3uR*q2Q1R + flux3vR*q3Q1R
     &             + flux3wR*q4Q1R + flux3pR*q5Q1R
!
        dfm(3,2) = flux3rR*q1Q2R + flux3uR*q2Q2R + flux3vR*q3Q2R
     &             + flux3wR*q4Q2R + flux3pR*q5Q2R
!
        dfm(3,3) = flux3rR*q1Q3R + flux3uR*q2Q3R + flux3vR*q3Q3R
     &             + flux3wR*q4Q3R + flux3pR*q5Q3R
!
        dfm(3,4) = flux3rR*q1Q4R + flux3uR*q2Q4R + flux3vR*q3Q4R
     &             + flux3wR*q4Q4R + flux3pR*q5Q4R
!
        dfm(3,5) = flux3rR*q1Q5R + flux3uR*q2Q5R + flux3vR*q3Q5R
     &             + flux3wR*q4Q5R + flux3pR*q5Q5R
!
!
        dfm(4,1) = flux4rR*q1Q1R + flux4uR*q2Q1R + flux4vR*q3Q1R
     &             + flux4wR*q4Q1R + flux4pR*q5Q1R
!
        dfm(4,2) = flux4rR*q1Q2R + flux4uR*q2Q2R + flux4vR*q3Q2R
     &             + flux4wR*q4Q2R + flux4pR*q5Q2R
!
        dfm(4,3) = flux4rR*q1Q3R + flux4uR*q2Q3R + flux4vR*q3Q3R
     &             + flux4wR*q4Q3R + flux4pR*q5Q3R
!
        dfm(4,4) = flux4rR*q1Q4R + flux4uR*q2Q4R + flux4vR*q3Q4R
     &             + flux4wR*q4Q4R + flux4pR*q5Q4R
!
        dfm(4,5) = flux4rR*q1Q5R + flux4uR*q2Q5R + flux4vR*q3Q5R
     &             + flux4wR*q4Q5R + flux4pR*q5Q5R
!
!
        dfm(5,1) = flux5rR*q1Q1R + flux5uR*q2Q1R + flux5vR*q3Q1R
     &             + flux5wR*q4Q1R + flux5pR*q5Q1R
!
        dfm(5,2) = flux5rR*q1Q2R + flux5uR*q2Q2R + flux5vR*q3Q2R
     &             + flux5wR*q4Q2R + flux5pR*q5Q2R
!
        dfm(5,3) = flux5rR*q1Q3R + flux5uR*q2Q3R + flux5vR*q3Q3R
     &             + flux5wR*q4Q3R + flux5pR*q5Q3R
!
        dfm(5,4) = flux5rR*q1Q4R + flux5uR*q2Q4R + flux5vR*q3Q4R
     &             + flux5wR*q4Q4R + flux5pR*q5Q4R
!
        dfm(5,5) = flux5rR*q1Q5R + flux5uR*q2Q5R + flux5vR*q3Q5R
     &             + flux5wR*q4Q5R + flux5pR*q5Q5R
        flops = flops + 4446.0
!
! Now take care of contribution to node 1
!
!       idiag = iau(node1)
!
! Diagonal piece
!
       if (node1 .le. nnodes) then
!       do j = 1,5
!        irow(j) = 5*(node1-1)+j-1
!        icol(j) = irow(j)
!        do k = 1,5
!         in = k + 10*(j-1)
!         val(in) = dfp(j,k)
!        enddo
!       enddo
!       call MatSetValuesLocal(A,5,irow,5,irow,val,ADD_VALUES,ierr)
!       call CHK_ERR(irank,ierr,irow(1),irow(1))

!
!       A(idiag,1,1) = A(idiag,1,1) + dfp(1,1)
!       A(idiag,1,2) = A(idiag,1,2) + dfp(1,2)
!       A(idiag,1,3) = A(idiag,1,3) + dfp(1,3)
!       A(idiag,1,4) = A(idiag,1,4) + dfp(1,4)
!
!       A(idiag,2,1) = A(idiag,2,1) + dfp(2,1)
!       A(idiag,2,2) = A(idiag,2,2) + dfp(2,2)
!       A(idiag,2,3) = A(idiag,2,3) + dfp(2,3)
!       A(idiag,2,4) = A(idiag,2,4) + dfp(2,4)
!
!       A(idiag,3,1) = A(idiag,3,1) + dfp(3,1)
!       A(idiag,3,2) = A(idiag,3,2) + dfp(3,2)
!       A(idiag,3,3) = A(idiag,3,3) + dfp(3,3)
!       A(idiag,3,4) = A(idiag,3,4) + dfp(3,4)
!
!       A(idiag,4,1) = A(idiag,4,1) + dfp(4,1)
!       A(idiag,4,2) = A(idiag,4,2) + dfp(4,2)
!       A(idiag,4,3) = A(idiag,4,3) + dfp(4,3)
!       A(idiag,4,4) = A(idiag,4,4) + dfp(4,4)
!
!       A(idiag,5,1) = A(idiag,5,1) + dfp(5,1)
!       A(idiag,5,2) = A(idiag,5,2) + dfp(5,2)
!       A(idiag,5,3) = A(idiag,5,3) + dfp(5,3)
!       A(idiag,5,4) = A(idiag,5,4) + dfp(5,4)
!       A(idiag,5,5) = A(idiag,5,5) + dfp(5,5)
!       do j = 1,5
!        icol(5+j) = 5*(node2-1)+j-1
!        do k = 1,5
!         in = k + 5 + 10*(j-1)
!         val(in) = dfm(j,k)
!        enddo
!       enddo
#if defined(INTERLACING)
#if defined(BLOCKING)
        irow(1) = node1 - 1
        icol(1) = node1 - 1
        icol(2) = node2 - 1
        call MatSetValuesBlockedLocal(A,1,irow,2,icol,
     >                                val1,ADD_VALUES,ierr)
#else
        do j = 1,5
         irow(j) = 5*(node1-1)+j-1
         icol(j) = irow(j)
         icol(5+j) = 5*(node2-1)+j-1
        enddo
        call MatSetValuesLocal(A,5,irow,10,icol,val1,ADD_VALUES,ierr)
#endif
#else
        do j = 1,5
         irow(j) = (node1-1)+(j-1)*nnodes
         icol(j) = irow(j)
         icol(5+j) = (node2-1)+(j-1)*nnodes
        enddo
        call MatSetValues(A,5,irow,10,icol,val1,ADD_VALUES,ierr)
#endif
       endif

!
! Now grab the offdiagonal
!
!       ioff = fhelp(n,1)
!       A(ioff,1,1) = A(ioff,1,1) + dfm(1,1)
!       A(ioff,1,2) = A(ioff,1,2) + dfm(1,2)
!       A(ioff,1,3) = A(ioff,1,3) + dfm(1,3)
!       A(ioff,1,4) = A(ioff,1,4) + dfm(1,4)
!
!       A(ioff,2,1) = A(ioff,2,1) + dfm(2,1)
!       A(ioff,2,2) = A(ioff,2,2) + dfm(2,2)
!       A(ioff,2,3) = A(ioff,2,3) + dfm(2,3)
!       A(ioff,2,4) = A(ioff,2,4) + dfm(2,4)
!
!       A(ioff,3,1) = A(ioff,3,1) + dfm(3,1)
!       A(ioff,3,2) = A(ioff,3,2) + dfm(3,2)
!       A(ioff,3,3) = A(ioff,3,3) + dfm(3,3)
!       A(ioff,3,4) = A(ioff,3,4) + dfm(3,4)
!
!       A(ioff,4,1) = A(ioff,4,1) + dfm(4,1)
!       A(ioff,4,2) = A(ioff,4,2) + dfm(4,2)
!       A(ioff,4,3) = A(ioff,4,3) + dfm(4,3)
!       A(ioff,4,4) = A(ioff,4,4) + dfm(4,4)
!
!       A(ioff,5,1) = A(ioff,5,1) + dfm(5,1)
!       A(ioff,5,2) = A(ioff,5,2) + dfm(5,2)
!       A(ioff,5,3) = A(ioff,5,3) + dfm(5,3)
!       A(ioff,5,4) = A(ioff,5,4) + dfm(5,4)
!       A(ioff,5,5) = A(ioff,5,5) + dfm(5,5)
!
! Now do the second node
!
       if (node2 .le. nnodes) then
!       do j = 1,5
!        irow(j) = 5*(node2-1)+j-1
!        icol(j) = irow(j)
!        do k = 1,5
!         in = k + 10*(j-1)
!         val(in) = -dfm(j,k)
!        enddo
!       enddo
!       call MatSetValuesLocal(A,5,irow,5,irow,val,ADD_VALUES,ierr)
!       call CHK_ERR(irank,ierr,irow(1),irow(1))

!       idiag = iau(node2)
!       A(idiag,1,1) = A(idiag,1,1) - dfm(1,1)
!       A(idiag,1,2) = A(idiag,1,2) - dfm(1,2)
!       A(idiag,1,3) = A(idiag,1,3) - dfm(1,3)
!       A(idiag,1,4) = A(idiag,1,4) - dfm(1,4)
!
!       A(idiag,2,1) = A(idiag,2,1) - dfm(2,1)
!       A(idiag,2,2) = A(idiag,2,2) - dfm(2,2)
!       A(idiag,2,3) = A(idiag,2,3) - dfm(2,3)
!       A(idiag,2,4) = A(idiag,2,4) - dfm(2,4)
!
!       A(idiag,3,1) = A(idiag,3,1) - dfm(3,1)
!       A(idiag,3,2) = A(idiag,3,2) - dfm(3,2)
!       A(idiag,3,3) = A(idiag,3,3) - dfm(3,3)
!       A(idiag,3,4) = A(idiag,3,4) - dfm(3,4)
!
!       A(idiag,4,1) = A(idiag,4,1) - dfm(4,1)
!       A(idiag,4,2) = A(idiag,4,2) - dfm(4,2)
!       A(idiag,4,3) = A(idiag,4,3) - dfm(4,3)
!       A(idiag,4,4) = A(idiag,4,4) - dfm(4,4)
!
!       A(ioff,5,1) = A(ioff,5,1) - dfp(5,1)
!       A(ioff,5,2) = A(ioff,5,2) - dfp(5,2)
!       A(ioff,5,3) = A(ioff,5,3) - dfp(5,3)
!       A(ioff,5,4) = A(ioff,5,4) - dfp(5,4)
!       A(ioff,5,5) = A(ioff,5,5) - dfp(5,5)
!
!       do j = 1,5
!        icol(5+j) = 5*(node1-1)+j-1
!        do k = 1,5
!         in = k + 5 + 10*(j-1)
!         val(in) = -dfp(j,k)
!        enddo
!       enddo
! Exchange elements in place
        do j = 1,5
         do k = 1,10
!         temp = -val1(k,j)
!         val1(k,j) = -val1(k+5,j)
!         val1(k+5,j) = temp
          val1(k,j) = -val1(k,j)
         enddo
        enddo
!
!       call CHK_ERR(irank,ierr,irow(1),icol(1))
#if defined(INTERLACING)
#if defined(BLOCKING)
        irow(1) = node2 - 1
        icol(1) = node1 - 1
        icol(2) = node2 - 1
        call MatSetValuesBlockedLocal(A,1,irow,2,icol,
     >                         val1,ADD_VALUES,ierr)
#else
        do j = 1,5
         irow(j) = 5*(node2-1)+j-1
         icol(j) = 5*(node1-1)+j-1
         icol(5+j) = irow(j)
        enddo
        call MatSetValuesLocal(A,5,irow,10,icol,val1,ADD_VALUES,ierr)
#endif
#else
        do j = 1,5
         irow(j) = (node2-1)+(j-1)*nnodes
         icol(j) = (node1-1)+(j-1)*nnodes
         icol(5+j) = irow(j)
        enddo
        call MatSetValues(A,5,irow,10,icol,val1,ADD_VALUES,ierr)
#endif
      endif

!
! Now grab the offdiagonal
!
!       ioff = fhelp(n,2)
!       A(ioff,1,1) = A(ioff,1,1) - dfp(1,1)
!       A(ioff,1,2) = A(ioff,1,2) - dfp(1,2)
!       A(ioff,1,3) = A(ioff,1,3) - dfp(1,3)
!       A(ioff,1,4) = A(ioff,1,4) - dfp(1,4)
!
!       A(ioff,2,1) = A(ioff,2,1) - dfp(2,1)
!       A(ioff,2,2) = A(ioff,2,2) - dfp(2,2)
!       A(ioff,2,3) = A(ioff,2,3) - dfp(2,3)
!       A(ioff,2,4) = A(ioff,2,4) - dfp(2,4)
!
!       A(ioff,3,1) = A(ioff,3,1) - dfp(3,1)
!       A(ioff,3,2) = A(ioff,3,2) - dfp(3,2)
!       A(ioff,3,3) = A(ioff,3,3) - dfp(3,3)
!       A(ioff,3,4) = A(ioff,3,4) - dfp(3,4)
!
!       A(ioff,4,1) = A(ioff,4,1) - dfp(4,1)
!       A(ioff,4,2) = A(ioff,4,2) - dfp(4,2)
!       A(ioff,4,3) = A(ioff,4,3) - dfp(4,3)
!       A(ioff,4,4) = A(ioff,4,4) - dfp(4,4)

        endif
 1030 continue
!
!  Convert back to conserved variables
!
      call PTOE(nvertices,qvec)
!
! Now we have to close the boundaries
! Use this if you want the 5/6-1/6 weighting
!
!     fivesix = 5./6.
!     onesix  = 1./6.
!     do 1040 i = 1,nsface
!       iface = isface(i)
!       node1 = eptr(iface,1)
!       node2 = eptr(iface,2)
!
!       rho1 = qnode(1,node1)
!       u1   = qnode(2,node1)/rho1
!       v1   = qnode(3,node1)/rho1
!       w1   = qnode(4,node1)/rhol
!       e1   = qnode(5,node1)
!       q21  = u1*u1 + v1*v1 + w1*w1
!
!       rho2 = qnode(1,node2)
!       u2   = qnode(2,node2)/rho2
!       v2   = qnode(3,node2)/rho2
!       w2   = qnode(4,node2)/rho2
!       e2   = qnode(5,node2)
!       q22  = u2*u2 + v2*v2 + w2*w2
!
!       xmean = .5*(x(node1) + x(node2))
!       ymean = .5*(y(node1) + y(node2))
!       zmean = .5*(z(node1) + z(node2))
!
!       xnorm =   ymean - y(node1)
!       ynorm = -(xmean - x(node1))
!       rlen  = sqrt(xnorm*xnorm + ynorm*ynorm)
!       xnorm = xnorm/rlen
!       ynorm = ynorm/rlen
!
!      idiag = iau(node1)
!      ioff  = fhelp(iface,1)
!
!      constx = fivesix*rlen*gm1*xnorm
!      consty = fivesix*rlen*gm1*ynorm
!
!      A(idiag,2,1) = A(idiag,2,1) + constx*q21/2.
!      A(idiag,2,2) = A(idiag,2,2) - constx*u1
!      A(idiag,2,3) = A(idiag,2,3) - constx*v1
!      A(idiag,2,4) = A(idiag,2,4) + constx
!
!      A(idiag,3,1) = A(idiag,3,1) + consty*q21/2.
!      A(idiag,3,2) = A(idiag,3,2) - consty*u1
!      A(idiag,3,3) = A(idiag,3,3) - consty*v1
!      A(idiag,3,4) = A(idiag,3,4) + consty
!
! Offdiagonal term
!
!      constx = onesix*rlen*gm1*xnorm
!      consty = onesix*rlen*gm1*ynorm
!
!      A(ioff,2,1) = A(ioff,2,1) + constx*q22/2.
!      A(ioff,2,2) = A(ioff,2,2) - constx*u2
!      A(ioff,2,3) = A(ioff,2,3) - constx*v2
!      A(ioff,2,4) = A(ioff,2,4) + constx
!
!      A(ioff,3,1) = A(ioff,3,1) + consty*q22/2.
!      A(ioff,3,2) = A(ioff,3,2) - consty*u2
!      A(ioff,3,3) = A(ioff,3,3) - consty*v2
!      A(ioff,3,4) = A(ioff,3,4) + consty
!
! Second node
!
!      idiag = iau(node2)
!      ioff  = fhelp(iface,2)
!
!      constx = fivesix*rlen*gm1*xnorm
!      consty = fivesix*rlen*gm1*ynorm
!
!      A(idiag,2,1) = A(idiag,2,1) + constx*q22/2.
!      A(idiag,2,2) = A(idiag,2,2) - constx*u2
!      A(idiag,2,3) = A(idiag,2,3) - constx*v2
!      A(idiag,2,4) = A(idiag,2,4) + constx
!
!      A(idiag,3,1) = A(idiag,3,1) + consty*q22/2.
!      A(idiag,3,2) = A(idiag,3,2) - consty*u2
!      A(idiag,3,3) = A(idiag,3,3) - consty*v2
!      A(idiag,3,4) = A(idiag,3,4) + consty
!
! Offdiagonal term
!
!      constx = onesix*rlen*gm1*xnorm
!      consty = onesix*rlen*gm1*ynorm
!
!      A(ioff,2,1) = A(ioff,2,1) + constx*q21/2.
!      A(ioff,2,2) = A(ioff,2,2) - constx*u1
!      A(ioff,2,3) = A(ioff,2,3) - constx*v1
!      A(ioff,2,4) = A(ioff,2,4) + constx
!
!      A(ioff,3,1) = A(ioff,3,1) + consty*q21/2.
!      A(ioff,3,2) = A(ioff,3,2) - consty*u1
!      A(ioff,3,3) = A(ioff,3,3) - consty*v1
!      A(ioff,3,4) = A(ioff,3,4) + consty
!
!1040 continue
!
! Use this piece if you dont use the 5/6-1/6 weighting
!
      do 1080 i = 1,nsnode
        inode = isnode(i)
        if (inode .le. nnodes) then
         xnorm = sxn(i)
         ynorm = syn(i)
         znorm = szn(i)
         rlen  = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
         xnorm = xnorm/rlen
         ynorm = ynorm/rlen
         znorm = znorm/rlen
!
         rho = qnode(1,inode)
         u   = qnode(2,inode)/rho
         v   = qnode(3,inode)/rho
         w   = qnode(4,inode)/rho
         e   = qnode(5,inode)
         q2  = u*u + v*v + w*w
!
         constx = rlen*xnorm*gm1
         consty = rlen*ynorm*gm1
         constz = rlen*znorm*gm1
!
         val(1) = constx*q2/2.0
         val(2) = - constx*u
         val(3) = - constx*v
         val(4) = - constx*w
         val(5) = constx
!
         val(6) = consty*q2/2.0
         val(7) = - consty*u
         val(8) = - consty*v
         val(9) = - consty*w
         val(10) = consty
!
         val(11) = constz*q2/2.0
         val(12) = - constz*u
         val(13) = - constz*v
         val(14) = - constz*w
         val(15) = constz
!
#if defined(INTERLACING)
         irow(1) = 5*(inode-1) + 1
         irow(2) = 5*(inode-1) + 2
         irow(3) = 5*(inode-1) + 3
         icol(1) = 5*(inode - 1)
         icol(2) = 5*(inode-1) + 1
         icol(3) = 5*(inode-1) + 2
         icol(4) = 5*(inode-1) + 3
         icol(5) = 5*(inode-1) + 4
        call MatSetValuesLocal(A,3,irow,5,icol,val,ADD_VALUES,ierr)
#else
         irow(1) = inode - 1 + nnodes
         irow(2) = inode - 1 + nnodes*2
         irow(3) = inode - 1 + nnodes*3
         icol(1) = inode - 1
         icol(2) = inode - 1 + nnodes
         icol(3) = inode - 1 + nnodes*2
         icol(4) = inode - 1 + nnodes*3
         icol(5) = inode - 1 + nnodes*4
         call MatSetValues(A,3,irow,5,icol,val,ADD_VALUES,ierr)
#endif
         flops = flops + 47.0
        endif

!
!        idiag = iau(inode)
!        A(idiag,2,1) = A(idiag,2,1) + constx*q2/2.
!        A(idiag,2,2) = A(idiag,2,2) - constx*u
!        A(idiag,2,3) = A(idiag,2,3) - constx*v
!        A(idiag,2,4) = A(idiag,2,4) - constx*w
!        A(idiag,2,5) = A(idiag,2,5) + constx
!
!        A(idiag,3,1) = A(idiag,3,1) + consty*q2/2.
!        A(idiag,3,2) = A(idiag,3,2) - consty*u
!        A(idiag,3,3) = A(idiag,3,3) - consty*v
!        A(idiag,3,4) = A(idiag,3,4) - consty*w
!        A(idiag,3,5) = A(idiag,3,5) + consty
!
!        A(idiag,4,1) = A(idiag,4,1) + constz*q2/2.
!        A(idiag,4,2) = A(idiag,4,2) - constz*u
!        A(idiag,4,3) = A(idiag,4,3) - constz*v
!        A(idiag,4,4) = A(idiag,4,4) - constz*w
!        A(idiag,4,5) = A(idiag,4,5) + constz
 1080 continue
!     print *, "Finished doing inviscid nodes"
!
! Now do viscous faces
!
      Prandtl = 0.72
      Twall = 1.0
      Twall = 1.0 + .5*sqrt(Prandtl)*gm1*xmach*xmach
      const = twall/ggm1
!
!     do 1050 i = 1,nvnode
!       inode = ivnode(i)
!       idiag = iau(inode)
!
! First zero out all the ones on the row and then fill them in
!
!       jstart = ia(inode)
!       jend   = ia(inode+1) - 1
!
!       do 1060 j = jstart,jend
!
! If this isn't the diagonal zero it out
! (This way we dont disturb the row for the continuity equation
!
!         if (j.ne.idiag) then
!          A(j,1,1) = 0.0
!          A(j,1,2) = 0.0
!          A(j,1,3) = 0.0
!          A(j,1,4) = 0.0
!          A(j,1,5) = 0.0
!
!          A(j,2,1) = 0.0
!          A(j,2,2) = 0.0
!          A(j,2,3) = 0.0
!          A(j,2,4) = 0.0
!          A(j,2,5) = 0.0
!
!          A(j,3,1) = 0.0
!          A(j,3,2) = 0.0
!          A(j,3,3) = 0.0
!          A(j,3,4) = 0.0
!          A(j,3,5) = 0.0
!
!          A(j,4,1) = 0.0
!          A(j,4,2) = 0.0
!          A(j,4,3) = 0.0
!          A(j,4,4) = 0.0
!          A(j,4,5) = 0.0
!
!          A(j,5,1) = 0.0
!          A(j,5,2) = 0.0
!          A(j,5,3) = 0.0
!          A(j,5,4) = 0.0
!          A(j,5,5) = 0.0
!
!         end if
!1060   continue
!
! Now set the diagonal
!
!       A(idiag,2,1) = 0.0
!       A(idiag,2,2) = 1.0
!       A(idiag,2,3) = 0.0
!       A(idiag,2,4) = 0.0
!       A(idiag,2,5) = 0.0
!
!       A(idiag,3,1) = 0.0
!       A(idiag,3,2) = 0.0
!       A(idiag,3,3) = 1.0
!       A(idiag,3,4) = 0.0
!       A(idiag,3,5) = 0.0
!
!       A(idiag,4,1) = 0.0
!       A(idiag,4,2) = 0.0
!       A(idiag,4,3) = 0.0
!       A(idiag,4,4) = 1.0
!       A(idiag,4,5) = 0.0
!
!       A(idiag,5,1) = -const
!       A(idiag,5,2) = 0.0
!       A(idiag,5,3) = 0.0
!       A(idiag,5,4) = 0.0
!       A(idiag,5,5) = 1.0
!
!1050 continue
!
! Now do the farfield
!
      s0 = c0*c0/(gamma*rho0**gm1)
      xgm1 = 1./gm1
      flops = flops + 5.0
      do 1070 i = 1,nfnode
        inode  = ifnode(i)
        if (inode .le. nnodes) then
        xnorm = fxn(i)
        ynorm = fyn(i)
        znorm = fzn(i)
        rlen  = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
        xnorm = xnorm/rlen
        ynorm = ynorm/rlen
        znorm = znorm/rlen
!
        rhoi = qnode(1,inode)
        ui   = qnode(2,inode)/rhoi
        vi   = qnode(3,inode)/rhoi
        wi   = qnode(4,inode)/rhoi
        ei   = qnode(5,inode)
        pi   = gm1*(ei - .5*rhoi*(ui*ui + vi*vi + wi*wi))
!
        unormi = xnorm*ui + ynorm*vi + znorm*wi
        unormo = xnorm*u0 + ynorm*v0 + znorm*w0
        ci    = sqrt(gamma*pi/rhoi)
!
        rplus  = unormi + 2.*ci/gm1
        rminus = unormo - 2.*c0/gm1
        unormb = .5*(rplus + rminus)
        cb     = .25*gm1*(rplus - rminus)
!
! If unormb > 0 this is outflow: take entropy from inside
! If unormb < 0 this is inflow: take entropy from outside
!
        con = .5*gamma*gm1/(rhoi*ci)
        cir =  con*(ui*ui + vi*vi + wi*wi - ei/rhoi)
        cim = -con*ui
        cin = -con*vi
        cio = -con*wi
        cie =  con
!
        pir = .5*gm1*(ui*ui + vi*vi + wi*wi)
        pim = -gm1*ui
        pin = -gm1*vi
        pio = -gm1*wi
        pie =  gm1
!
        unormir = -unormi/rhoi
        unormim = xnorm/rhoi
        unormin = ynorm/rhoi
        unormio = znorm/rhoi
        unormie = 0.0
!
        rplusr = unormir + 2.*cir/gm1
        rplusm = unormim + 2.*cim/gm1
        rplusn = unormin + 2.*cin/gm1
        rpluso = unormio + 2.*cio/gm1
        rpluse = unormie + 2.*cie/gm1
!
        rminusr = 0.0
        rminusm = 0.0
        rminusn = 0.0
        rminuso = 0.0
        rminuse = 0.0
!
        unormbr = .5*(rplusr + rminusr)
        unormbm = .5*(rplusm + rminusm)
        unormbn = .5*(rplusn + rminusn)
        unormbo = .5*(rpluso + rminuso)
        unormbe = .5*(rpluse + rminuse)
!
        cbr = .25*gm1*(rplusr - rminusr)
        cbm = .25*gm1*(rplusm - rminusm)
        cbn = .25*gm1*(rplusn - rminusn)
        cbo = .25*gm1*(rpluso - rminuso)
        cbe = .25*gm1*(rpluse - rminuse)
        flops = flops + 121.0
!
        if (unormb .gt. 0.0) then
         ub = ui + xnorm*(unormb - unormi)
         vb = vi + ynorm*(unormb - unormi)
         wb = wi + znorm*(unormb - unormi)
         sb    = ci*ci/(gamma*rhoi**gm1)
         rhob  = (cb*cb/(gamma*sb))**xgm1
         q2b   = ub*ub + vb*vb + wb*wb
         pb    = rhob*cb*cb/gamma
         eb    = pb/gm1 + .5*rhob*q2b
!
! Some derivatives
!
         ubr = -ui/rhoi + xnorm*(unormbr - unormir)
         ubm = 1./rhoi  + xnorm*(unormbm - unormim)
         ubn =            xnorm*(unormbn - unormin)
         ubo =            xnorm*(unormbo - unormio)
         ube =            xnorm*(unormbe - unormie)
!
         vbr = -vi/rhoi + ynorm*(unormbr - unormir)
         vbm =            ynorm*(unormbm - unormim)
         vbn = 1./rhoi  + ynorm*(unormbn - unormin)
         vbo =            ynorm*(unormbo - unormio)
         vbe =            ynorm*(unormbe - unormie)
!
         wbr = -wi/rhoi + znorm*(unormbr - unormir)
         wbm =            znorm*(unormbm - unormim)
         wbn =            znorm*(unormbn - unormin)
         wbo = 1./rhoi  + znorm*(unormbo - unormio)
         wbe =            znorm*(unormbe - unormie)
!
         sbr = (pir - ci*ci)/(rhoi**gamma)
         sbm = pim/(rhoi**gamma)
         sbn = pin/(rhoi**gamma)
         sbo = pio/(rhoi**gamma)
         sbe = pie/(rhoi**gamma)
!
         con   = rhob/(cb*gm1)
         rhobr = con*(2.*cbr - cb/sb*sbr)
         rhobm = con*(2.*cbm - cb/sb*sbm)
         rhobn = con*(2.*cbn - cb/sb*sbn)
         rhobo = con*(2.*cbo - cb/sb*sbo)
         rhobe = con*(2.*cbe - cb/sb*sbe)
!
         pbr = (2.*rhob*cb*cbr + cb*cb*rhobr)/gamma
         pbm = (2.*rhob*cb*cbm + cb*cb*rhobm)/gamma
         pbn = (2.*rhob*cb*cbn + cb*cb*rhobn)/gamma
         pbo = (2.*rhob*cb*cbo + cb*cb*rhobo)/gamma
         pbe = (2.*rhob*cb*cbe + cb*cb*rhobe)/gamma
!
!234567890c234567890c234567890c234567890c234567890c234567890c23456789012
         ebr = pbr/gm1 + .5*(2.*rhob*(ub*ubr + vb*vbr + wb*wbr)
     >         + q2b*rhobr)
         ebm = pbm/gm1 + .5*(2.*rhob*(ub*ubm + vb*vbm + wb*wbm)
     >         + q2b*rhobm)
         ebn = pbn/gm1 + .5*(2.*rhob*(ub*ubn + vb*vbn + wb*wbn)
     >         + q2b*rhobn)
         ebo = pbo/gm1 + .5*(2.*rhob*(ub*ubo + vb*vbo + wb*wbo)
     >         + q2b*rhobo)
         ebe = pbe/gm1 + .5*(2.*rhob*(ub*ube + vb*vbe + wb*wbe)
     >         + q2b*rhobe)
         flops = flops + 208.0
!
        else
         ub = u0 + xnorm*(unormb - unormo)
         vb = v0 + ynorm*(unormb - unormo)
         wb = w0 + znorm*(unormb - unormo)
         sb = s0
         rhob  = (cb*cb/(gamma*sb))**xgm1
         q2b   = ub*ub + vb*vb + wb*wb
         pb    = rhob*cb*cb/gamma
         eb    = pb/gm1 + .5*rhob*q2b
!
! Some derivatives
!
         unormor = 0.0
         unormom = 0.0
         unormon = 0.0
         unormoo = 0.0
         unormoe = 0.0
!
         ubr = xnorm*(unormbr - unormor)
         ubm = xnorm*(unormbm - unormom)
         ubn = xnorm*(unormbn - unormon)
         ubo = xnorm*(unormbo - unormoo)
         ube = xnorm*(unormbe - unormoe)
!
         vbr = ynorm*(unormbr - unormor)
         vbm = ynorm*(unormbm - unormom)
         vbn = ynorm*(unormbn - unormon)
         vbo = ynorm*(unormbo - unormoo)
         vbe = ynorm*(unormbe - unormoe)
!
         wbr = znorm*(unormbr - unormor)
         wbm = znorm*(unormbm - unormom)
         wbn = znorm*(unormbn - unormon)
         wbo = znorm*(unormbo - unormoo)
         wbe = znorm*(unormbe - unormoe)
!
         sbr = 0.0
         sbm = 0.0
         sbn = 0.0
         sbo = 0.0
         sbe = 0.0
!
         con   = rhob/(cb*gm1)
         rhobr = con*(2.*cbr - cb/sb*sbr)
         rhobm = con*(2.*cbm - cb/sb*sbm)
         rhobn = con*(2.*cbn - cb/sb*sbn)
         rhobo = con*(2.*cbo - cb/sb*sbo)
         rhobe = con*(2.*cbe - cb/sb*sbe)
!
         pbr = (2.*rhob*cb*cbr + cb*cb*rhobr)/gamma
         pbm = (2.*rhob*cb*cbm + cb*cb*rhobm)/gamma
         pbn = (2.*rhob*cb*cbn + cb*cb*rhobn)/gamma
         pbo = (2.*rhob*cb*cbo + cb*cb*rhobo)/gamma
         pbe = (2.*rhob*cb*cbe + cb*cb*rhobe)/gamma
!
         ebr = pbr/gm1 + .5*(2.*rhob*(ub*ubr + vb*vbr + wb*wbr)
     >         + q2b*rhobr)
         ebm = pbm/gm1 + .5*(2.*rhob*(ub*ubm + vb*vbm + wb*wbm)
     >         + q2b*rhobm)
         ebn = pbn/gm1 + .5*(2.*rhob*(ub*ubn + vb*vbn + wb*wbn)
     >         + q2b*rhobn)
         ebo = pbo/gm1 + .5*(2.*rhob*(ub*ubo + vb*vbo + wb*wbo)
     >         + q2b*rhobo)
         ebe = pbe/gm1 + .5*(2.*rhob*(ub*ube + vb*vbe + wb*wbe)
     >         + q2b*rhobe)
         flops = flops + 177.0
!
        end if
!
! Now add contribution to lhs
!

         val(1) = rlen*(rhob*unormbr + unormb*rhobr)
         val(2) = rlen*(rhob*unormbm + unormb*rhobm)
         val(3) = rlen*(rhob*unormbn + unormb*rhobn)
         val(4) = rlen*(rhob*unormbo + unormb*rhobo)
         val(5) = rlen*(rhob*unormbe + unormb*rhobe)
!
         val(6) = rlen*(rhob*ub*unormbr
     1             + unormb*(rhob*ubr + ub*rhobr) + xnorm*pbr)
         val(7) = rlen*(rhob*ub*unormbm
     1             + unormb*(rhob*ubm + ub*rhobm) + xnorm*pbm)
         val(8) = rlen*(rhob*ub*unormbn
     1             + unormb*(rhob*ubn + ub*rhobn) + xnorm*pbn)
         val(9) = rlen*(rhob*ub*unormbo
     1             + unormb*(rhob*ubo + ub*rhobo) + xnorm*pbo)
         val(10) = rlen*(rhob*ub*unormbe
     1             + unormb*(rhob*ube + ub*rhobe) + xnorm*pbe)
!
         val(11) = rlen*(rhob*vb*unormbr
     1             + unormb*(rhob*vbr + vb*rhobr) + ynorm*pbr)
         val(12) = rlen*(rhob*vb*unormbm
     1             + unormb*(rhob*vbm + vb*rhobm) + ynorm*pbm)
         val(13) = rlen*(rhob*vb*unormbn
     1             + unormb*(rhob*vbn + vb*rhobn) + ynorm*pbn)
         val(14) = rlen*(rhob*vb*unormbo
     1             + unormb*(rhob*vbo + vb*rhobo) + ynorm*pbo)
         val(15) = rlen*(rhob*vb*unormbe
     1             + unormb*(rhob*vbe + vb*rhobe) + ynorm*pbe)
!
         val(16) = rlen*(rhob*wb*unormbr
     1             + unormb*(rhob*wbr + wb*rhobr) + znorm*pbr)
         val(17) = rlen*(rhob*wb*unormbm
     1             + unormb*(rhob*wbm + wb*rhobm) + znorm*pbm)
         val(18) = rlen*(rhob*wb*unormbn
     1             + unormb*(rhob*wbn + wb*rhobn) + znorm*pbn)
         val(19) = rlen*(rhob*wb*unormbo
     1             + unormb*(rhob*wbo + wb*rhobo) + znorm*pbo)
         val(20) = rlen*(rhob*wb*unormbe
     1             + unormb*(rhob*wbe + wb*rhobe) + znorm*pbe)
!
         val(21) = rlen*((eb + pb)*unormbr
     1                            + unormb*(ebr + pbr))
         val(22) = rlen*((eb + pb)*unormbm
     1                            + unormb*(ebm + pbm))
         val(23) = rlen*((eb + pb)*unormbn
     1                            + unormb*(ebn + pbn))
         val(24) = rlen*((eb + pb)*unormbo
     1                            + unormb*(ebo + pbo))
         val(25) = rlen*((eb + pb)*unormbe
     1                            + unormb*(ebe + pbe))
!
#if defined(INTERLACING)
#if defined(BLOCKING)
         irow(1) = inode - 1
         call MatSetValuesBlockedLocal(A,1,irow,1,irow,
     >                                 val,ADD_VALUES,ierr)
#else
         do k = 1,5
          irow(k) = 5*(inode-1)+k-1
         enddo
         call MatSetValuesLocal(A,5,irow,5,irow,val,ADD_VALUES,ierr)
#endif
#else
         do k = 1,5
          irow(k) = inode - 1 + nnodes*(k-1)
         enddo
         call MatSetValues(A,5,irow,5,irow,val,ADD_VALUES,ierr)
#endif

         flops = flops + 220.0
        endif
!
 1070 continue

!     print *, "Finished doing far field nodes"

!  Assemble matrix
      call PetscLogFlops(flops,ierr)

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)
!     call MatView(A, VIEWER_STDOUT_SELF,ierr)
      flag = SAME_NONZERO_PATTERN
!
! End of subroutine FILLA
!
      return
      end
!
!

      subroutine CHK_ERR(irank, ierr, irow, icol)
      if (ierr .gt. 0) then
       write(*,*) 'On processor ',irank, ': Non-zero entry in row ',
     1 irow, ' and column ',icol,' is beyond the pre-allocated memory'
      endif
      return
      end



