c
c  This file contains code for approximating the Jacobian matrix
c  with sparse finite differences.
c
c ---------------------------------------------------------------

      integer function jformdt2(eps,epsi,ltog,nloc,mat,is1,
     &                 b1bc,b2bc,b3bc,b2bc_tmp,d,
     &                 dt,xx,p,xx_bc,p_bc,
     &                 br,bl,be,sadai,sadaj,sadak,
     &                 aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                 f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2,fff,ao)
c
c jformdt2 - Forms the Jacobian matrix, J(x), with direct assembly 
c            of the matrix into the PETSc formats via calls to
c            MatSetValues().  This version uses implicit boundary
c            conditions.  This approach of direct matrix assembly 
c            is recommended -- rather than the use of the intermediate
c            Eagle format as in the routine jformdt().  This routine
c            is called by FormJacobian().
c
#include "param.h"
#include "dt.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "freq.h"
#include "ang.h"
#include "es.h"
#include "bc.h"
#include "diag.h"
#include "bcimpl.h"
#include "fv.h"
#include "sigma.h"
#include "ibcwork.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
c      COMMON /DVRBLS/ DR(NI1,NJ1,NK1),DRU(NI1,NJ1,NK1),DRV(NI1,NJ1,NK1)
c      COMMON /DVRBLS/ DRW(NI1,NJ1,NK1),DE(NI1,NJ1,NK1)
c      COMMON /M/ B1(5,5,NI,NJ,NK),B2(5,5,NI,NJ,NK),B3(5,5,NI,NJ,NK)
c      COMMON /M/ B4(5,5,NI,NJ,NK),B5(5,5,NI,NJ,NK),B6(5,5,NI,NJ,NK)
c      COMMON /DIAG/ D(5,5,NI,NJ,NK)
c      COMMON /TSTEP/ DT(NI,NJ,NK)
c      COMMON /ANG/ ALPHA,BETA,PHI
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /FREQ/ IFREQ,ISGS
c      COMMON /BCINFO/  b1bc(5,5,4,nj,nk), b2bc(5,5,ni,4,nk),
c     &                 b3bc(5,5,ni,nj,4), b2bc_tmp(5,5,ni,4,nk)

      double precision  qer(5),qel(5),eps,epsi,dti
      integer m,ierr,nloc,ltog(0:nloc-1),nd2
      integer is1(*),set_block,rcblock
      integer ijk,jk,rc(5),i,j,k,ndm,ndem,ndof_euler
      Mat     mat
      AO      ao
      jformdt2 = 0

      ndof_euler = 5
      do 11 m=1,5
         qel(m)=zero
         qer(m)=zero
 11   continue

      do 30 m=1,5
      qel(m)=eps
      ierr = ndm (m,epsi,qer,qel,b1bc,b2bc,b3bc,b2bc_tmp,
     &          mat,ltog,nloc,is1,
     &          xx,p,xx_bc,p_bc,fff,
     &          sadai,sadaj,sadak,
     &          aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,ao)

      if (ierr .ne. 0) then
         jformdt2 = -1
         return 
      endif
      ierr = ndem (m,epsi,qer,qel,b1bc,b2bc,b3bc,b2bc_tmp,
     &          mat,ltog,nloc,
     &          xx,p,fff,
     &          sadai,sadaj,sadak,
     &          aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
      if (ierr .ne. 0) then
         jformdt2 = -1
         return 
      endif

      ierr = nd2 (m,epsi,qer,qel,1,mat,ltog,nloc,d,
     &          xx,p,
     &          br,bl,be,sadai,sadaj,sadak,
     &          aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &          f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)

      if (ierr .ne. 0) then
         jformdt2 = -1
         return 
      endif
      qel(m)=zero
   30 continue

      do 31 m=1,5
      qer(m)=eps
      ierr = nd2 (m,epsi,qer,qel,2,mat,ltog,nloc,d,
     &          xx,p,
     &          br,bl,be,sadai,sadaj,sadak,
     &          aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &          f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
      if (ierr .ne. 0) then
         jformdt2 = -1
         return 
      endif
      qer(m)=zero
   31 continue
c Find eigendecomposition of diagonal block of Jacobian; Form DT array.
c But really need to recalculate dt only when the iterates change. 
      call eigenv(dt,xx,p,
     &            sadai,sadaj,sadak,
     &            aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)

c We must specify that we are inserting matrix blocks by columns, since
c the PETSc default is row-oriented input.
      call MatSetOption(mat,MAT_COLUMN_ORIENTED,ierr)

C We're not currently using set_block
      set_block = 0
      do 21 k=zsf2,zef01
      do 21 j=ysf2,yef01
       jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
      do 21 i=xsf2,xef01
c      dql=B1(L,M,I,J,K)+B2(L,M,I,J,K)+B3(L,M,I,J,K)
c      dqr=B4(L,M,I-1,J,K)+B5(L,M,I,J-1,K)+B6(L,M,I,J,K-1)
c      D(L,M,I,J,K)=dql-dqr
c      IF (L.EQ.M) D(L,M,I,J,K)=D(L,M,I,J,K)+one/DT(I,J,K)
       ijk = ltog(ndof * (i-gxsf + jk))
       rc(1) = ijk
       rc(2) = ijk + 1
       rc(3) = ijk + 2
       rc(4) = ijk + 3
       rc(5) = ijk + 4
       dti = one/DT(I,J,K)
       D(1,1,I,J,K)=D(1,1,I,J,K) + dti
       D(2,2,I,J,K)=D(2,2,I,J,K) + dti
       D(3,3,I,J,K)=D(3,3,I,J,K) + dti
       D(4,4,I,J,K)=D(4,4,I,J,K) + dti
       D(5,5,I,J,K)=D(5,5,I,J,K) + dti
cc       do 21 m=1,5
cc          D(m,m,I,J,K)=D(m,m,I,J,K) + dti
cc          call MatSetValues(mat,ndof,rc,1,rc(m),d(1,m,i,j,k),
cc     &                      INSERT_VALUES,ierr)
       if (set_block .eq. 1) then
          rcblock = rc(1)/ndof
          call MatSetValuesBlocked(mat,1,rcblock,1,rcblock,
     &                             d(1,1,i,j,k),INSERT_VALUES,ierr)
       else
          call MatSetValues(mat,ndof_euler,rc,ndof_euler,rc,
     &                      d(1,1,i,j,k),INSERT_VALUES,ierr)
       endif
   21 CONTINUE

C  The rest of the input is row-oriented.
      call MatSetOption(mat,MAT_ROW_ORIENTED,ierr)

      return
      end

c ---------------------------------------------------------------
c
      integer function ndm (m,epsi,qer,qel,b1bc,b2bc,b3bc,b2bc_tmp,
     &                mat,ltog,nloc,is1,
     &                xx,p,xx_bc,p_bc,fff,
     &                sadai,sadaj,sadak,
     &                aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,ao)
c
c ndm - Uses finite differencing to compute a numerical Jacobian
c       estimation for j=1 boundary grid points only (all other 
c       boundary points are handled in the routine ndem()).  Interior
c       grid points are handled separately in nd() or nd2().
c
c       This routine can either store the matrix in an intermediate
c       work array for later transition to a PETSc format (if nloc = -1),
c       or assemble the PETSc matrix directly via calls to 
c       MatSetValues() (more efficient).
c
c       This routine is called by jformdt() (using the intermediate
c       storage arrays) and jformdt2() (direct PETSc matrix assembly).
c
c       This routine used as a starting point similar uniprocessor code
c       written by M.D. Tidriri.
c
c Input Parameters:
c    mat      - matrix data structure
c    ltog     - local-to-global mapping for distributed arrays
c    nloc     - number of elements in ltog mapping
c    is1      - index set with PETSc ordering numbers for certain
c               boundary nodes

#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "init.h"
#include "bc.h"
#include "bcimpl.h"
#include "ibcwork.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /SADA/ SADAI(NI,NJ,NK),SADAJ(NI,NJ,NK),SADAK(NI,NJ,NK)
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /INIT/ RINIT,RUINIT,RVINIT,RWINIT,EINIT

      double precision  fb1rj1,fb1ruj1,fb1rvj1,fb1rwj1,fb1ej1
      double precision  r1,ru1,rv1,epsi,rw1,e1,r2,ru2,rv2
      double precision  rw2,r0,pi1k,pb,w,rqdq
      double precision  e2,ada,p0,p1,p2,c0,ff,qda,u,v
      double precision  qer(5),qel(5),bw(10)
      integer i,j,k,m,row(5),jk,IV,id,ijkid,ijkn,col(2)
      integer nloc,ltog(0:nloc-1),ijk,ierr,is1(0:*),ict
      Mat     mat
      AO      ao

c     IV corresponds to INSERT_VALUES mode for MatSetValues
      IV = INSERT_VALUES
c
      ndm = 0
      if (ysf1 .eq. 1) then
        j=1
        ict=m-1
        do 2 k=zsf2,zef01
           jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
        do 2 i=xsf2,xef01
           ijk = ndof * (i-gxsf + jk) - 1
           ijkn = ijk + ndof*gxm
c      DO 2 K=2,NK
c      DO 2 I=2,NI
      IF (K.GT.KTIP) GO TO 24
      IF (I.LE.ITL.OR.I.GT.ITU) GO TO 24
      r1=R(I,1,K)+qel(1)
      ru1=RU(I,1,K)+qel(2)
      rv1=RV(I,1,K)+qel(3)
      rw1=RW(I,1,K)+qel(4)
      e1=E(I,1,K)+qel(5)
      r2=R(I,2,K)
      ru2=RU(I,2,K)
      rv2=RV(I,2,K)
      rw2=RW(I,2,K)
      e2=E(I,2,K)
      p1=gm1*(e1-p5*(ru1**2+rv1**2+rw1**2)
     * /r1)
      p2=gm1*(e2-p5*(ru2**2+rv2**2+rw2**2)
     * /r2)
      IF (NCYC.GE.bcswitch) GO TO 15
      ADA=SADAJ(I,1,K)**2
      QDA=RU2*AJX(I,1,K)+RV2*AJY(I,1,K)+RW2*AJZ(I,1,K)
      FF=two*QDA/ADA
      fb1ruj1=ru1-RU2+FF*AJX(I,1,K)
      fb1rvj1=rv1-RV2+FF*AJY(I,1,K)
      fb1rwj1=rw1-RW2+FF*AJZ(I,1,K)
c      fb1rj1(i,k)=r1-R2
      fb1rj1=r1-R2
      fb1ej1=e1-E2
c     Diagonal term
      if (nloc .eq. -1) then
         b2bc(1,m,i,1,k)=(fb1rj1-fbcrj1(i,k))*epsi
         b2bc(2,m,i,1,k)=(fb1ruj1-fbcruj1(i,k))*epsi
         b2bc(3,m,i,1,k)=(fb1rvj1-fbcrvj1(i,k))*epsi
         b2bc(4,m,i,1,k)=(fb1rwj1-fbcrwj1(i,k))*epsi
         b2bc(5,m,i,1,k)=(fb1ej1-fbcej1(i,k))*epsi
      else
c         bw(1)=(fb1rj1-fbcrj1(i,k))*epsi
c         bw(2)=(fb1ruj1-fbcruj1(i,k))*epsi
c         bw(3)=(fb1rvj1-fbcrvj1(i,k))*epsi
c         bw(4)=(fb1rwj1-fbcrwj1(i,k))*epsi
c         bw(5)=(fb1ej1-fbcej1(i,k))*epsi
         bw(1)=(fb1rj1-fbcrj1(i,k))*epsi
         bw(3)=(fb1ruj1-fbcruj1(i,k))*epsi
         bw(5)=(fb1rvj1-fbcrvj1(i,k))*epsi
         bw(7)=(fb1rwj1-fbcrwj1(i,k))*epsi
         bw(9)=(fb1ej1-fbcej1(i,k))*epsi
      endif
c
      r1=R(I,1,K)
      ru1=RU(I,1,K)
      rv1=RV(I,1,K)
      rw1=RW(I,1,K)
      e1=E(I,1,K)
      r2=R(I,2,K)+qel(1)
      ru2=RU(I,2,K)+qel(2)
      rv2=RV(I,2,K)+qel(3)
      rw2=RW(I,2,K)+qel(4)
      e2=E(I,2,K)+qel(5)
      p1=gm1*(e1-p5*(ru1**2+rv1**2+rw1**2)
     * /r1)
      p2=gm1*(e2-p5*(ru2**2+rv2**2+rw2**2)
     * /r2)
c
      ADA=SADAJ(I,1,K)**2
      QDA=RU2*AJX(I,1,K)+RV2*AJY(I,1,K)+RW2*AJZ(I,1,K)
      FF=two*QDA/ADA
      fb1ruj1=ru1-RU2+FF*AJX(I,1,K)
      fb1rvj1=rv1-RV2+FF*AJY(I,1,K)
      fb1rwj1=rw1-RW2+FF*AJZ(I,1,K)
      fb1rj1=r1-R2
c      fb1rj1(i,k)=r1-R2
      fb1ej1=e1-E2
c     Northern neighbor
      if (nloc .eq. -1) then
         b2bc(1,m,i,2,k)=(fb1rj1-fbcrj1(i,k))*epsi
         b2bc(2,m,i,2,k)=(fb1ruj1-fbcruj1(i,k))*epsi
         b2bc(3,m,i,2,k)=(fb1rvj1-fbcrvj1(i,k))*epsi
         b2bc(4,m,i,2,k)=(fb1rwj1-fbcrwj1(i,k))*epsi
         b2bc(5,m,i,2,k)=(fb1ej1-fbcej1(i,k))*epsi
      else
         row(1) = ltog(ijk+1)
         row(2) = row(1) + 1
         row(3) = row(2) + 1
         row(4) = row(3) + 1
         row(5) = row(4) + 1
         col(1) = ltog(ijk+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
c         bw(1)=(fb1rj1-fbcrj1(i,k))*epsi
c         bw(2)=(fb1ruj1-fbcruj1(i,k))*epsi
c         bw(3)=(fb1rvj1-fbcrvj1(i,k))*epsi
c         bw(4)=(fb1rwj1-fbcrwj1(i,k))*epsi
c         bw(5)=(fb1ej1-fbcej1(i,k))*epsi
         bw(2)=(fb1rj1-fbcrj1(i,k))*epsi
         bw(4)=(fb1ruj1-fbcruj1(i,k))*epsi
         bw(6)=(fb1rvj1-fbcrvj1(i,k))*epsi
         bw(8)=(fb1rwj1-fbcrwj1(i,k))*epsi
         bw(10)=(fb1ej1-fbcej1(i,k))*epsi
c         col(1) = ltog(ijkn+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
         col(2) = ltog(ijkn+m)
         call MatSetValues(mat,5,row,2,col,bw,IV,ierr)
      endif
      GO TO 2
   15 CONTINUE
C   THESE ARE CHARACTERISTIC VARIABLE IMPERMEABLE WALL B.C.s
      QDA=(RU2*AJX(I,1,K)+RV2*AJY(I,1,K)
     *   +RW2*AJZ(I,1,K))/R2
      P0=p5*(P1+P2)
      R0=p5*(R1+R2)
      C0=SQRT(GAM*P0/R0)
      PB=P2-R0*C0*QDA/SADAJ(I,1,K)
      PI1K=2.*PB-P2
      fb1rj1=r1-R2-2.*(PB-P2)/C0**2
c      fb1rj1(i,k)=r1-R2-2.*(PB-P2)/C0**2
      FF=two*QDA/SADAJ(I,1,K)**2
      U=RU2/R2-FF*AJX(I,1,K)
      V=RV2/R2-FF*AJY(I,1,K)
      W=RW2/R2-FF*AJZ(I,1,K)
      fb1ruj1=ru1-U*R1
      fb1rvj1=rv1-V*R1
      fb1rwj1=rw1-W*R1
      RQDQ=p5*(RU1**2+RV1**2+RW1**2)/R1
      fb1ej1=e1-PI1K*RGM1-RQDQ
c     Diagonal term (j=1)
      if (nloc .eq. -1) then
         b2bc(1,m,i,1,k)=(fb1rj1-fbcrj1(i,k))*epsi
         b2bc(2,m,i,1,k)=(fb1ruj1-fbcruj1(i,k))*epsi
         b2bc(3,m,i,1,k)=(fb1rvj1-fbcrvj1(i,k))*epsi
         b2bc(4,m,i,1,k)=(fb1rwj1-fbcrwj1(i,k))*epsi
         b2bc(5,m,i,1,k)=(fb1ej1-fbcej1(i,k))*epsi
      else
c         bw(1)=(fb1rj1-fbcrj1(i,k))*epsi
c         bw(2)=(fb1ruj1-fbcruj1(i,k))*epsi
c         bw(3)=(fb1rvj1-fbcrvj1(i,k))*epsi
c         bw(4)=(fb1rwj1-fbcrwj1(i,k))*epsi
c         bw(5)=(fb1ej1-fbcej1(i,k))*epsi
         bw(1)=(fb1rj1-fbcrj1(i,k))*epsi
         bw(3)=(fb1ruj1-fbcruj1(i,k))*epsi
         bw(5)=(fb1rvj1-fbcrvj1(i,k))*epsi
         bw(7)=(fb1rwj1-fbcrwj1(i,k))*epsi
         bw(9)=(fb1ej1-fbcej1(i,k))*epsi
      endif
c
      r1=R(I,1,K)
      ru1=RU(I,1,K)
      rv1=RV(I,1,K)
      rw1=RW(I,1,K)
      e1=E(I,1,K)
      r2=R(I,2,K)+qel(1)
      ru2=RU(I,2,K)+qel(2)
      rv2=RV(I,2,K)+qel(3)
      rw2=RW(I,2,K)+qel(4)
      e2=E(I,2,K)+qel(5)
      p1=gm1*(e1-p5*(ru1**2+rv1**2+rw1**2)
     * /r1)
      p2=gm1*(e2-p5*(ru2**2+rv2**2+rw2**2)
     * /r2)
      QDA=(RU2*AJX(I,1,K)+RV2*AJY(I,1,K)
     *   +RW2*AJZ(I,1,K))/R2
      P0=p5*(P1+P2)
      R0=p5*(R1+R2)
      C0=SQRT(GAM*P0/R0)
      PB=P2-R0*C0*QDA/SADAJ(I,1,K)
      PI1K=2.*PB-P2
      fb1rj1=r1-R2-2.*(PB-P2)/C0**2
c      fb1rj1(i,k)=r1-R2-2.*(PB-P2)/C0**2
      FF=two*QDA/SADAJ(I,1,K)**2
      U=RU2/R2-FF*AJX(I,1,K)
      V=RV2/R2-FF*AJY(I,1,K)
      W=RW2/R2-FF*AJZ(I,1,K)
      fb1ruj1=ru1-U*R1
      fb1rvj1=rv1-V*R1
      fb1rwj1=rw1-W*R1
      RQDQ=p5*(RU1**2+RV1**2+RW1**2)/R1
      fb1ej1=e1-PI1K*RGM1-RQDQ
c     Northern neighbor (j=2)
      if (nloc .eq. -1) then
         b2bc(1,m,i,2,k)=(fb1rj1-fbcrj1(i,k))*epsi
         b2bc(2,m,i,2,k)=(fb1ruj1-fbcruj1(i,k))*epsi
         b2bc(3,m,i,2,k)=(fb1rvj1-fbcrvj1(i,k))*epsi
         b2bc(4,m,i,2,k)=(fb1rwj1-fbcrwj1(i,k))*epsi
         b2bc(5,m,i,2,k)=(fb1ej1-fbcej1(i,k))*epsi
      else
         row(1) = ltog(ijk+1)
         row(2) = row(1) + 1
         row(3) = row(2) + 1
         row(4) = row(3) + 1
         row(5) = row(4) + 1
         col(1) = ltog(ijk+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
c         bw(1)=(fb1rj1-fbcrj1(i,k))*epsi
c         bw(2)=(fb1ruj1-fbcruj1(i,k))*epsi
c         bw(3)=(fb1rvj1-fbcrvj1(i,k))*epsi
c         bw(4)=(fb1rwj1-fbcrwj1(i,k))*epsi
c         bw(5)=(fb1ej1-fbcej1(i,k))*epsi
         bw(2)=(fb1rj1-fbcrj1(i,k))*epsi
         bw(4)=(fb1ruj1-fbcruj1(i,k))*epsi
         bw(6)=(fb1rvj1-fbcrvj1(i,k))*epsi
         bw(8)=(fb1rwj1-fbcrwj1(i,k))*epsi
         bw(10)=(fb1ej1-fbcej1(i,k))*epsi
c         col(1) = ltog(ijkn+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
         col(2) = ltog(ijkn+m)
         call MatSetValues(mat,5,row,2,col,bw,IV,ierr)
      endif
c
      goto 2
 24   continue
      ID=NI+1-I
c      ID=NI+2-I
c Note: This j value corresponds to j=2!!
      ijkid = ndof * ( id-1 + j*ni1 + (k-1)*ni1*nj1 ) - 1
      r1=R(I,1,K)+qel(1)
      ru1=RU(I,1,K)+qel(2)
      rv1=RV(I,1,K)+qel(3)
      rw1=RW(I,1,K)+qel(4)
      e1=E(I,1,K)+qel(5)
      r2=R_bc(I,2,K)
      ru2=RU_bc(I,2,K)
      rv2=RV_bc(I,2,K)
      rw2=RW_bc(I,2,K)
      e2=E_bc(I,2,K)
c      r2=R(ID,2,K)
c      ru2=RU(ID,2,K)
c      rv2=RV(ID,2,K)
c      rw2=RW(ID,2,K)
c      e2=E(ID,2,K)
      p1=gm1*(e1-p5*(ru1**2+rv1**2+rw1**2)
     * /r1)
      p2=gm1*(e2-p5*(ru2**2+rv2**2+rw2**2)
     * /r2)
      fb1ruj1=ru1-RU2
      fb1rvj1=rv1-RV2
      fb1rwj1=rw1-RW2
      fb1rj1=r1-R2
c      fb1rj1(i,k)=r1-R2
      fb1ej1=e1-E2
c     Diagonal term (j=1)
      if (nloc .eq. -1) then
         b2bc(1,m,i,1,k)=(fb1rj1-fbcrj1(i,k))*epsi
         b2bc(2,m,i,1,k)=(fb1ruj1-fbcruj1(i,k))*epsi
         b2bc(3,m,i,1,k)=(fb1rvj1-fbcrvj1(i,k))*epsi
         b2bc(4,m,i,1,k)=(fb1rwj1-fbcrwj1(i,k))*epsi
         b2bc(5,m,i,1,k)=(fb1ej1-fbcej1(i,k))*epsi
      else
         bw(1)=(fb1rj1-fbcrj1(i,k))*epsi
         bw(2)=(fb1ruj1-fbcruj1(i,k))*epsi
         bw(3)=(fb1rvj1-fbcrvj1(i,k))*epsi
         bw(4)=(fb1rwj1-fbcrwj1(i,k))*epsi
         bw(5)=(fb1ej1-fbcej1(i,k))*epsi
         row(1) = ltog(ijk+1)
         row(2) = row(1) + 1
         row(3) = row(2) + 1
         row(4) = row(3) + 1
         row(5) = row(4) + 1
         col(1) = ltog(ijk+m)
         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
      endif
c
      r1=R(I,1,K)
      ru1=RU(I,1,K)
      rv1=RV(I,1,K)
      rw1=RW(I,1,K)
      e1=E(I,1,K)
      r2=R_bc(I,2,K)+qel(1)
      ru2=RU_bc(I,2,K)+qel(2)
      rv2=RV_bc(I,2,K)+qel(3)
      rw2=RW_bc(I,2,K)+qel(4)
      e2=E_bc(I,2,K)+qel(5)
c      r2=R(ID,2,K)+qel(1)
c      ru2=RU(ID,2,K)+qel(2)
c      rv2=RV(ID,2,K)+qel(3)
c      rw2=RW(ID,2,K)+qel(4)
c      e2=E(ID,2,K)+qel(5)
      p1=gm1*(e1-p5*(ru1**2+rv1**2+rw1**2)
     * /r1)
      p2=gm1*(e2-p5*(ru2**2+rv2**2+rw2**2)
     * /r2)
c
      fb1ruj1=ru1-RU2
      fb1rvj1=rv1-RV2
      fb1rwj1=rw1-RW2
      fb1rj1=r1-R2
c      fb1rj1(i,k)=r1-R2
      fb1ej1=e1-E2
c     Northern neighbor (j=2)
      if (nloc .eq. -1) then
         b2bc_tmp(1,m,i,2,k)=(fb1rj1-fbcrj1(i,k))*epsi
         b2bc_tmp(2,m,i,2,k)=(fb1ruj1-fbcruj1(i,k))*epsi
         b2bc_tmp(3,m,i,2,k)=(fb1rvj1-fbcrvj1(i,k))*epsi
         b2bc_tmp(4,m,i,2,k)=(fb1rwj1-fbcrwj1(i,k))*epsi
         b2bc_tmp(5,m,i,2,k)=(fb1ej1-fbcej1(i,k))*epsi
      else
c         row(1) = ltog(ijk+1)
c         row(2) = row(1) + 1
c         row(3) = row(2) + 1
c         row(4) = row(3) + 1
c         row(5) = row(4) + 1
         bw(1)=(fb1rj1-fbcrj1(i,k))*epsi
         bw(2)=(fb1ruj1-fbcruj1(i,k))*epsi
         bw(3)=(fb1rvj1-fbcrvj1(i,k))*epsi
         bw(4)=(fb1rwj1-fbcrwj1(i,k))*epsi
         bw(5)=(fb1ej1-fbcej1(i,k))*epsi
c
c Note: We cannot call AOApplicationToPetsc() here, since this
c       parallel database routine MUST be called by all processors.
c       Luckily, we have already determined these column numbers
c       when setting up the vector scatters.
c         uniprocessor: col(1) = ijkid+m
         col(1) = ijkid+m
         call AOApplicationToPetsc(ao,1,col(1),ierr)
cc         col(1) = is1(ict)
         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
      endif
      ict = ict + ndof
c      b2bc(1,m,id,2,k)=(fb1rj1-Fbcrj1(i,k))*epsi
c      b2bc(2,m,id,2,k)=(fb1ruj1-fbcruj1(i,k))*epsi
c      b2bc(3,m,id,2,k)=(fb1rvj1-fbcrvj1(i,k))*epsi
c      b2bc(4,m,id,2,k)=(fb1rwj1-fbcrwj1(i,k))*epsi
c      b2bc(5,m,id,2,k)=(fb1ej1-fbcej1(i,k))*epsi
c
    2 CONTINUE
      endif
c
      RETURN
      END
c ---------------------------------------------------------------

      integer function ndem (m,epsi,qer,qel,b1bc,b2bc,b3bc,b2bc_tmp,
     &                mat,ltog,nloc,
     &                xx,p,fff,
     &                sadai,sadaj,sadak,
     &                aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
c
c ndem - Uses finite differencing to compute a numerical Jacobian
c        estimation for boundary grid points only (all but the j=1
c        boundary, which is handled in the routine ndem()).  Interior
c        grid points are handled separately in nd() or nd2().
c
c        This routine can either store the matrix in an intermediate
c        work array for later transition to a PETSc format (if nloc = -1),
c        or assemble the PETSc matrix directly via calls to 
c        MatSetValues() (more efficient).
c
c        This routine is called by jformdt() (using the intermediate
c        storage arrays) and jformdt2() (direct PETSc matrix assembly).
c
c        This routine used as a starting point similar uniprocessor code
c        written by M.D. Tidriri.

#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "init.h"
#include "bcimpl.h"
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
c      double precision b1bc(5,5,4,nj,nk),b2bc(5,5,ni,4,nk)
c      double precision b3bc(5,5,ni,nj,4)

      double precision  fb1ri1,fb1rui1,fb1rvi1,fb1rwi1,fb1ei1
      double precision  fb1rk1,fb1ruk1,fb1rvk1,fb1rwk1,fb1ek1
      double precision  fb1ri2,fb1rui2,fb1rvi2,fb1rwi2,fb1ei2
      double precision  fb1rj2,fb1ruj2,fb1rvj2,fb1rwj2,fb1ej2
      double precision  fb1rk2,fb1ruk2,fb1rvk2,fb1rwk2,fb1ek2

      double precision  qer(5),qel(5),epsi,rnj1,runj1
      double precision  rvnj1,rwnj1,rnj,runj,rvnj,rwnj,enj,p0,pnj
      double precision  pnj1,c0,con,r0c0,u,ub,u2,v,vb,v2,w,wb,w2,r1,ru1
      double precision  rv1,rw1,e1,r2,ru2,rv2,rw2,e2,ada,pij1,p2,rnk1
      double precision  runk1,rvnk1,rwnk1,enk1,rnk,runk,rvnk,enk,pijnk1
      double precision  pnk,enj1,r0,pb,pinj1k,rb,p1,qda,ff,rwnk,pnk1
      double precision  rni1,runi1,rvni1,rwni1,eni1,rni,runi,rvni,rwni
      double precision  eni,r1jk,rinj1k,pni,pni1,rni1jk,bw(10)
      integer i,j,k,m,row(5),col(2),IV,ik,ijk,ijkn,jk
      integer nloc,ltog(0:nloc-1),ierr
      Mat     mat

c     IV corresponds to INSERT_VALUES mode for MatSetValues
      IV = INSERT_VALUES
c
      ndem = 0
      if (yef01 .eq. nj) then
        j=nj1
        do 20 k=zsf2,zef01
           jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
        do 20 i=xsf2,xef01
            ijk  = ndof * (i-gxsf + jk) - 1
            ijkn = ijk - ndof*gxm
c      DO 20 K=2,NK
c      DO 20 I=2,NI
C   THESE ARE CHARACTERISTIC VARIABLE SUBSONIC INFLOW B.C.s
      rnj1=R(I,NJ1,K)+qel(1)
      runj1=RU(I,NJ1,K)+qel(2)
      rvnj1=RV(I,NJ1,K)+qel(3)
      rwnj1=RW(I,NJ1,K)+qel(4)
      enj1=E(I,NJ1,K)+qel(5)
      rnj=R(I,NJ,K)
      runj=RU(I,NJ,K)
      rvnj=RV(I,NJ,K)
      rwnj=RW(I,NJ,K)
      enj=E(I,NJ,K)
      pnj1=gm1*(enj1-p5*(runj1**2+rvnj1**2+rwnj1**2)
     s /rnj1)
      pnj=gm1*(enj-p5*(runj**2+rvnj**2+rwnj**2)
     s /rnj)
      R0=p5*(RNJ+RNJ1)
      P0=p5*(PNJ+PNJ1)
      C0=SQRT(GAM*P0/R0)
      R0C0=R0*C0
      U2=RUNJ/RNJ
      V2=RVNJ/RNJ
      W2=RWNJ/RNJ
      PB=p5*(PINIT+PNJ-R0C0/SADAJ(I,NJ,K)*(AJX(I,NJ,K)*(RUINIT-
     s U2)+AJY(I,NJ,K)*(RVINIT-V2)+AJZ(I,NJ,K)*(RWINIT-W2)))
      RB=RINIT+(PB-PINIT)/C0**2
      CON=-(PINIT-PB)/(R0C0*SADAJ(I,NJ,K))
      UB=RUINIT+AJX(I,NJ,K)*CON
      VB=RVINIT+AJY(I,NJ,K)*CON
      WB=RWINIT+AJZ(I,NJ,K)*CON
      PINJ1K=2.*PB-PNJ
      RINJ1K=2.*RB-RNJ
      fb1rj2=RNJ1-2.*RB+RNJ
      U=2.*UB-U2
      V=2.*VB-V2
      W=2.*WB-W2
      fb1ruj2=RUNJ1-U*RINJ1K
      fb1rvj2=RVNJ1-V*RINJ1K
      fb1rwj2=RWNJ1-W*RINJ1K
      fb1ej2=ENJ1-PINJ1K*RGM1
     s                        -p5*RINJ1K*(U**2+V**2+W**2)
c     Diagonal term (j=nj1)
      if (nloc .eq. -1) then
         b2bc(1,m,i,4,k)=(fb1rj2-fbcrj1(i,k))*epsi
         b2bc(2,m,i,4,k)=(fb1ruj2-fbcruj1(i,k))*epsi
         b2bc(3,m,i,4,k)=(fb1rvj2-fbcrvj1(i,k))*epsi
         b2bc(4,m,i,4,k)=(fb1rwj2-fbcrwj1(i,k))*epsi
         b2bc(5,m,i,4,k)=(fb1ej2-fbcej1(i,k))*epsi
      else
c         bw(1)=(fb1rj2-fbcrj2(i,k))*epsi
c         bw(2)=(fb1ruj2-fbcruj2(i,k))*epsi
c         bw(3)=(fb1rvj2-fbcrvj2(i,k))*epsi
c         bw(4)=(fb1rwj2-fbcrwj2(i,k))*epsi
c         bw(5)=(fb1ej2-fbcej2(i,k))*epsi
         bw(2)=(fb1rj2-fbcrj2(i,k))*epsi
         bw(4)=(fb1ruj2-fbcruj2(i,k))*epsi
         bw(6)=(fb1rvj2-fbcrvj2(i,k))*epsi
         bw(8)=(fb1rwj2-fbcrwj2(i,k))*epsi
         bw(10)=(fb1ej2-fbcej2(i,k))*epsi
c         write(6,*) 'fff: ',fbcrj2(i,k),fbcruj2(i,k),
c     &        fbcrvj2(i,k),fbcrwj2(i,k),fbcej2(i,k)
c         write(6,*) 'bw: ',bw(2),bw(4),bw(6),bw(8),bw(10)
      endif
c
      rnj1=R(I,NJ1,K)
      runj1=RU(I,NJ1,K)
      rvnj1=RV(I,NJ1,K)
      rwnj1=RW(I,NJ1,K)
      enj1=E(I,NJ1,K)
      rnj=R(I,NJ,K)+qel(1)
      runj=RU(I,NJ,K)+qel(2)
      rvnj=RV(I,NJ,K)+qel(3)
      rwnj=RW(I,NJ,K)+qel(4)
      enj=E(I,NJ,K)+qel(5)
      pnj1=gm1*(enj1-p5*(runj1**2+rvnj1**2+rwnj1**2)
     * /rnj1)
      pnj=gm1*(enj-p5*(runj**2+rvnj**2+rwnj**2)
     * /rnj)
      R0=p5*(RNJ+RNJ1)
      P0=p5*(PNJ+PNJ1)
      C0=SQRT(GAM*P0/R0)
      R0C0=R0*C0
      U2=RUNJ/RNJ
      V2=RVNJ/RNJ
      W2=RWNJ/RNJ
      PB=p5*(PINIT+PNJ-R0C0/SADAJ(I,NJ,K)*(AJX(I,NJ,K)*(RUINIT-
     * U2)+AJY(I,NJ,K)*(RVINIT-V2)+AJZ(I,NJ,K)*(RWINIT-W2)))
      RB=RINIT+(PB-PINIT)/C0**2
      CON=-(PINIT-PB)/(R0C0*SADAJ(I,NJ,K))
      UB=RUINIT+AJX(I,NJ,K)*CON
      VB=RVINIT+AJY(I,NJ,K)*CON
      WB=RWINIT+AJZ(I,NJ,K)*CON
      PINJ1K=2.*PB-PNJ
      RINJ1K=2.*RB-RNJ
      fb1rj2=RNJ1-2.*RB+RNJ
      U=2.*UB-U2
      V=2.*VB-V2
      W=2.*WB-W2
      fb1ruj2=RUNJ1-U*RINJ1K
      fb1rvj2=RVNJ1-V*RINJ1K
      fb1rwj2=RWNJ1-W*RINJ1K
      fb1ej2=ENJ1-PINJ1K*RGM1
     s                        -p5*RINJ1K*(U**2+V**2+W**2)
c     Southern neighbor (j=nj)
      if (nloc .eq. -1) then
         b2bc(1,m,i,3,k)=(fb1rj2-fbcrj2(i,k))*epsi
         b2bc(2,m,i,3,k)=(fb1ruj2-fbcruj2(i,k))*epsi
         b2bc(3,m,i,3,k)=(fb1rvj2-fbcrvj2(i,k))*epsi
         b2bc(4,m,i,3,k)=(fb1rwj2-fbcrwj2(i,k))*epsi
         b2bc(5,m,i,3,k)=(fb1ej2-fbcej2(i,k))*epsi
      else
         row(1) = ltog(ijk+1)
         row(2) = row(1) + 1
         row(3) = row(2) + 1
         row(4) = row(3) + 1
         row(5) = row(4) + 1
         col(2) = ltog(ijk+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
c         bw(1)=(fb1rj2-fbcrj2(i,k))*epsi
c         bw(2)=(fb1ruj2-fbcruj2(i,k))*epsi
c         bw(3)=(fb1rvj2-fbcrvj2(i,k))*epsi
c         bw(4)=(fb1rwj2-fbcrwj2(i,k))*epsi
c         bw(5)=(fb1ej2-fbcej2(i,k))*epsi
         bw(1)=(fb1rj2-fbcrj2(i,k))*epsi
         bw(3)=(fb1ruj2-fbcruj2(i,k))*epsi
         bw(5)=(fb1rvj2-fbcrvj2(i,k))*epsi
         bw(7)=(fb1rwj2-fbcrwj2(i,k))*epsi
         bw(9)=(fb1ej2-fbcej2(i,k))*epsi
         col(1) = ltog(ijkn+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
         call MatSetValues(mat,5,row,2,col,bw,IV,ierr)
      endif
   20 CONTINUE
      endif
c
      if (zsf1 .eq. 1) then
        k=1
        do 6 j=ysf2,yef01
           jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
        do 6 i=xsf2,xef01
           ijk = ndof * (i-gxsf + jk) - 1
           ijkn = ijk + ndof*gxm*gym
c      DO 6 J=2,NJ
c      DO 6 I=2,NI
      r1=R(I,J,1)+qel(1)
      ru1=RU(I,J,1)+qel(2)
      rv1=RV(I,J,1)+qel(3)
      rw1=RW(I,J,1)+qel(4)
      e1=E(I,J,1)+qel(5)
      r2=R(I,J,2)
      ru2=RU(I,J,2)
      rv2=RV(I,J,2)
      rw2=RW(I,J,2)
      e2=E(I,J,2)
      p1=gm1*(e1-p5*(ru1**2+rv1**2+rw1**2)
     * /r1)
      p2=gm1*(e2-p5*(ru2**2+rv2**2+rw2**2)
     * /r2)
      ADA=AKX(I,J,1)**2+AKY(I,J,1)**2+AKZ(I,J,1)**2
      QDA=RU2*AKX(I,J,1)+RV2*AKY(I,J,1)
     *   +RW2*AKZ(I,J,1)
      fb1rk1=R1-R2
      fb1ek1=E1-E2
      PIJ1=P2
      FF=two*QDA/ADA
      fb1ruk1=RU1-RU2+FF*AKX(I,J,1)
      fb1rvk1=RV1-RV2+FF*AKY(I,J,1)
      fb1rwk1=RW1-RW2+FF*AKZ(I,J,1)
c     Diagonal term (k=1)
      if (nloc .eq. -1) then
         b3bc(1,m,i,j,1)=(fb1rk1-fbcrk1(i,j))*epsi
         b3bc(2,m,i,j,1)=(fb1ruk1-fbcruk1(i,j))*epsi
         b3bc(3,m,i,j,1)=(fb1rvk1-fbcrvk1(i,j))*epsi
         b3bc(4,m,i,j,1)=(fb1rwk1-fbcrwk1(i,j))*epsi
         b3bc(5,m,i,j,1)=(fb1ek1-fbcek1(i,j))*epsi
      else
c         bw(1)=(fb1rk1-fbcrk1(i,j))*epsi
c         bw(2)=(fb1ruk1-fbcruk1(i,j))*epsi
c         bw(3)=(fb1rvk1-fbcrvk1(i,j))*epsi
c         bw(4)=(fb1rwk1-fbcrwk1(i,j))*epsi
c         bw(5)=(fb1ek1-fbcek1(i,j))*epsi
         bw(1)=(fb1rk1-fbcrk1(i,j))*epsi
         bw(3)=(fb1ruk1-fbcruk1(i,j))*epsi
         bw(5)=(fb1rvk1-fbcrvk1(i,j))*epsi
         bw(7)=(fb1rwk1-fbcrwk1(i,j))*epsi
         bw(9)=(fb1ek1-fbcek1(i,j))*epsi
      endif
c
      r1=R(I,J,1)
      ru1=RU(I,J,1)
      rv1=RV(I,J,1)
      rw1=RW(I,J,1)
      e1=E(I,J,1)
      r2=R(I,J,2)+qel(1)
      ru2=RU(I,J,2)+qel(2)
      rv2=RV(I,J,2)+qel(3)
      rw2=RW(I,J,2)+qel(4)
      e2=E(I,J,2)+qel(5)
      p1=gm1*(e1-p5*(ru1**2+rv1**2+rw1**2)
     * /r1)
      p2=gm1*(e2-p5*(ru2**2+rv2**2+rw2**2)
     * /r2)
      ADA=AKX(I,J,1)**2+AKY(I,J,1)**2+AKZ(I,J,1)**2
      QDA=RU2*AKX(I,J,1)+RV2*AKY(I,J,1)
     *   +RW2*AKZ(I,J,1)
      fb1rk1=R1-R2
      fb1ek1=E1-E2
      PIJ1=P2
      FF=two*QDA/ADA
      fb1ruk1=RU1-RU2+FF*AKX(I,J,1)
      fb1rvk1=RV1-RV2+FF*AKY(I,J,1)
      fb1rwk1=RW1-RW2+FF*AKZ(I,J,1)
c     Upper neighbor (k=2)
      if (nloc .eq. -1) then
         b3bc(1,m,i,j,2)=(fb1rk1-fbcrk1(i,j))*epsi
         b3bc(2,m,i,j,2)=(fb1ruk1-fbcruk1(i,j))*epsi
         b3bc(3,m,i,j,2)=(fb1rvk1-fbcrvk1(i,j))*epsi
         b3bc(4,m,i,j,2)=(fb1rwk1-fbcrwk1(i,j))*epsi
         b3bc(5,m,i,j,2)=(fb1ek1-fbcek1(i,j))*epsi
      else
         row(1) = ltog(ijk+1)
         row(2) = row(1) + 1
         row(3) = row(2) + 1
         row(4) = row(3) + 1
         row(5) = row(4) + 1
         col(1) = ltog(ijk+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
c         bw(1)=(fb1rk1-fbcrk1(i,j))*epsi
c         bw(2)=(fb1ruk1-fbcruk1(i,j))*epsi
c         bw(3)=(fb1rvk1-fbcrvk1(i,j))*epsi
c         bw(4)=(fb1rwk1-fbcrwk1(i,j))*epsi
c         bw(5)=(fb1ek1-fbcek1(i,j))*epsi
         bw(2)=(fb1rk1-fbcrk1(i,j))*epsi
         bw(4)=(fb1ruk1-fbcruk1(i,j))*epsi
         bw(6)=(fb1rvk1-fbcrvk1(i,j))*epsi
         bw(8)=(fb1rwk1-fbcrwk1(i,j))*epsi
         bw(10)=(fb1ek1-fbcek1(i,j))*epsi
         col(2) = ltog(ijkn+m)
c         col(1) = ltog(ijkn+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
         call MatSetValues(mat,5,row,2,col,bw,IV,ierr)
      endif
    6 CONTINUE
      endif
c
      if (zef01 .eq. nk) then
        k=nk1
        do 7 j=ysf2,yef01
           jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
        do 7 i=xsf2,xef01
           ijk = ndof * (i-gxsf + jk) - 1
           ijkn = ijk - ndof*gxm*gym
      rnk1=R(I,J,NK1)+qel(1)
      runk1=RU(I,J,NK1)+qel(2)
      rvnk1=RV(I,J,NK1)+qel(3)
      rwnk1=RW(I,J,NK1)+qel(4)
      enk1=E(I,J,NK1)+qel(5)
      rnk=R(I,J,NK)
      runk=RU(I,J,NK)
      rvnk=RV(I,J,NK)
      rwnk=RW(I,J,NK)
      enk=E(I,J,NK)
      pnk1=gm1*(enk1-p5*(runk1**2+rvnk1**2+rwnk1**2)
     * /rnk1)
      pnk=gm1*(enk-p5*(runk**2+rvnk**2+rwnk**2)
     * /rnk)
      ADA=AKX(I,J,NK)**2+AKY(I,J,NK)**2+AKZ(I,J,NK)**2
      QDA=RUNK*AKX(I,J,NK)+RVNK*AKY(I,J,NK)
     *   +RWNK*AKZ(I,J,NK)
      fb1rk2=RNK1-RNK
      fb1ek2=ENK1-ENK
      PIJNK1=PNK
      FF=two*QDA/ADA
      fb1ruk2=RUNK1-RUNK+FF*AKX(I,J,NK)
      fb1rvk2=RVNK1-RVNK+FF*AKY(I,J,NK)
      fb1rwk2=RWNK1-RWNK+FF*AKZ(I,J,NK)
c     Diagonal term (k=nk1)
      if (nloc .eq. -1) then
         b3bc(1,m,i,j,4)=(fb1rk2-fbcrk2(i,j))*epsi
         b3bc(2,m,i,j,4)=(fb1ruk2-fbcruk2(i,j))*epsi
         b3bc(3,m,i,j,4)=(fb1rvk2-fbcrvk2(i,j))*epsi
         b3bc(4,m,i,j,4)=(fb1rwk2-fbcrwk2(i,j))*epsi
         b3bc(5,m,i,j,4)=(fb1ek2-fbcek2(i,j))*epsi
      else
c         bw(1)=(fb1rk2-fbcrk2(i,j))*epsi
c         bw(2)=(fb1ruk2-fbcruk2(i,j))*epsi
c         bw(3)=(fb1rvk2-fbcrvk2(i,j))*epsi
c         bw(4)=(fb1rwk2-fbcrwk2(i,j))*epsi
c         bw(5)=(fb1ek2-fbcek2(i,j))*epsi
         bw(2)=(fb1rk2-fbcrk2(i,j))*epsi
         bw(4)=(fb1ruk2-fbcruk2(i,j))*epsi
         bw(6)=(fb1rvk2-fbcrvk2(i,j))*epsi
         bw(8)=(fb1rwk2-fbcrwk2(i,j))*epsi
         bw(10)=(fb1ek2-fbcek2(i,j))*epsi
      endif
c
      rnk1=R(I,J,NK1)
      runk1=RU(I,J,NK1)
      rvnk1=RV(I,J,NK1)
      rwnk1=RW(I,J,NK1)
      enk1=E(I,J,NK1)
      rnk=R(I,J,NK)+qel(1)
      runk=RU(I,J,NK)+qel(2)
      rvnk=RV(I,J,NK)+qel(3)
      rwnk=RW(I,J,NK)+qel(4)
      enk=E(I,J,NK)+qel(5)
      pnk1=gm1*(enk1-p5*(runk1**2+rvnk1**2+rwnk1**2)
     * /rnk1)
      pnk=gm1*(enk-p5*(runk**2+rvnk**2+rwnk**2)
     * /rnk)
      ADA=AKX(I,J,NK)**2+AKY(I,J,NK)**2+AKZ(I,J,NK)**2
      QDA=RUNK*AKX(I,J,NK)+RVNK*AKY(I,J,NK)
     *   +RWNK*AKZ(I,J,NK)
      fb1rk2=RNK1-RNK
      fb1ek2=ENK1-ENK
      PIJNK1=PNK
      FF=two*QDA/ADA
      fb1ruk2=RUNK1-RUNK+FF*AKX(I,J,NK)
      fb1rvk2=RVNK1-RVNK+FF*AKY(I,J,NK)
      fb1rwk2=RWNK1-RWNK+FF*AKZ(I,J,NK)
c     Lower neighbor (k=nk)
      if (nloc .eq. -1) then
         b3bc(1,m,i,j,3)=(fb1rk2-fbcrk2(i,j))*epsi
         b3bc(2,m,i,j,3)=(fb1ruk2-fbcruk2(i,j))*epsi
         b3bc(3,m,i,j,3)=(fb1rvk2-fbcrvk2(i,j))*epsi
         b3bc(4,m,i,j,3)=(fb1rwk2-fbcrwk2(i,j))*epsi
         b3bc(5,m,i,j,3)=(fb1ek2-fbcek2(i,j))*epsi
      else
         row(1) = ltog(ijk+1)
         row(2) = row(1) + 1
         row(3) = row(2) + 1
         row(4) = row(3) + 1
         row(5) = row(4) + 1
         col(2) = ltog(ijk+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
c         bw(1)=(fb1rk2-fbcrk2(i,j))*epsi
c         bw(2)=(fb1ruk2-fbcruk2(i,j))*epsi
c         bw(3)=(fb1rvk2-fbcrvk2(i,j))*epsi
c         bw(4)=(fb1rwk2-fbcrwk2(i,j))*epsi
c         bw(5)=(fb1ek2-fbcek2(i,j))*epsi
         bw(1)=(fb1rk2-fbcrk2(i,j))*epsi
         bw(3)=(fb1ruk2-fbcruk2(i,j))*epsi
         bw(5)=(fb1rvk2-fbcrvk2(i,j))*epsi
         bw(7)=(fb1rwk2-fbcrwk2(i,j))*epsi
         bw(9)=(fb1ek2-fbcek2(i,j))*epsi
         col(1) = ltog(ijkn+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
         call MatSetValues(mat,5,row,2,col,bw,IV,ierr)
      endif
    7 CONTINUE
      endif
c
      if (xef01 .eq. ni) then
        i=ni1
        do 8 k=zsf2,zef01
           ik = i-gxsf + (k-gzsf)*gxm*gym
        do 8 j=ysf2,yef01
           ijk  = ndof * (ik + (j-gysf)*gxm) - 1
           ijkn = ijk - ndof
c      DO 8 K=2,NK
c      DO 8 J=2,NJ
C   THESE ARE CHARACTERISTIC VARIABLE SUBSONIC OUTFLOW B.C.s
      rni1=R(NI1,J,K)+qel(1)
      runi1=RU(NI1,J,K)+qel(2)
      rvni1=RV(NI1,J,K)+qel(3)
      rwni1=RW(NI1,J,K)+qel(4)
      eni1=E(NI1,J,K)+qel(5)
      rni=R(NI,J,K)
      runi=RU(NI,J,K)
      rvni=RV(NI,J,K)
      rwni=RW(NI,J,K)
      eni=E(NI,J,K)
      pni1=gm1*(eni1-p5*(runi1**2+rvni1**2+rwni1**2)
     * /rni1)
      pni=gm1*(eni-p5*(runi**2+rvni**2+rwni**2)
     * /rni)
      R0=p5*(RNI1+RNI)
      C0=SQRT(GAM*PINIT/R0)
      RNI1JK=RNI+2.*(PINIT-PNI)/C0**2
      fb1ri2=RNI1-RNI-2.*(PINIT-PNI)/C0**2
      CON=-two*(PINIT-PNI)/(SADAI(NI,J,K)*R0*C0)
      U=RUNI/RNI+AIX(NI,J,K)*CON
      V=RVNI/RNI+AIY(NI,J,K)*CON
      W=RWNI/RNI+AIZ(NI,J,K)*CON
      fb1rui2=RUNI1-U*RNI1JK
      fb1rvi2=RVNI1-V*RNI1JK
      fb1rwi2=RWNI1-W*RNI1JK
      fb1ei2=ENI1-PINIT*RGM1
     s                        -p5*RNI1JK*(U**2+V**2+W**2)
c     Diagonal term (i=ni1)
      if (nloc .eq. -1) then
         b1bc(1,m,4,j,k)=(fb1ri2-fbcri2(j,k))*epsi
         b1bc(2,m,4,j,k)=(fb1rui2-fbcrui2(j,k))*epsi
         b1bc(3,m,4,j,k)=(fb1rvi2-fbcrvi2(j,k))*epsi
         b1bc(4,m,4,j,k)=(fb1rwi2-fbcrwi2(j,k))*epsi
         b1bc(5,m,4,j,k)=(fb1ei2-fbcei2(j,k))*epsi
      else
c         bw(1)=(fb1ri2-fbcri2(j,k))*epsi
c         bw(2)=(fb1rui2-fbcrui2(j,k))*epsi
c         bw(3)=(fb1rvi2-fbcrvi2(j,k))*epsi
c         bw(4)=(fb1rwi2-fbcrwi2(j,k))*epsi
c         bw(5)=(fb1ei2-fbcei2(j,k))*epsi
         bw(2)=(fb1ri2-fbcri2(j,k))*epsi
         bw(4)=(fb1rui2-fbcrui2(j,k))*epsi
         bw(6)=(fb1rvi2-fbcrvi2(j,k))*epsi
         bw(8)=(fb1rwi2-fbcrwi2(j,k))*epsi
         bw(10)=(fb1ei2-fbcei2(j,k))*epsi
      endif
c
      rni1=R(NI1,J,K)
      runi1=RU(NI1,J,K)
      rvni1=RV(NI1,J,K)
      rwni1=RW(NI1,J,K)
      eni1=E(NI1,J,K)
      rni=R(NI,J,K)+qel(1)
      runi=RU(NI,J,K)+qel(2)
      rvni=RV(NI,J,K)+qel(3)
      rwni=RW(NI,J,K)+qel(4)
      eni=E(NI,J,K)+qel(5)
      pni1=gm1*(eni1-p5*(runi1**2+rvni1**2+rwni1**2)
     * /rni1)
      pni=gm1*(eni-p5*(runi**2+rvni**2+rwni**2)
     * /rni)
      R0=p5*(RNI1+RNI)
      C0=SQRT(GAM*PINIT/R0)
      RNI1JK=RNI+2.*(PINIT-PNI)/C0**2
      fb1ri2=RNI1-RNI-2.*(PINIT-PNI)/C0**2
      CON=-two*(PINIT-PNI)/(SADAI(NI,J,K)*R0*C0)
      U=RUNI/RNI+AIX(NI,J,K)*CON
      V=RVNI/RNI+AIY(NI,J,K)*CON
      W=RWNI/RNI+AIZ(NI,J,K)*CON
      fb1rui2=RUNI1-U*RNI1JK
      fb1rvi2=RVNI1-V*RNI1JK
      fb1rwi2=RWNI1-W*RNI1JK
      fb1ei2=ENI1-PINIT*RGM1
     s                        -p5*RNI1JK*(U**2+V**2+W**2)
c     Western neighbor (i=ni)
      if (nloc .eq. -1) then
         b1bc(1,m,3,j,k)=(fb1ri2-fbcri2(j,k))*epsi
         b1bc(2,m,3,j,k)=(fb1rui2-fbcrui2(j,k))*epsi
         b1bc(3,m,3,j,k)=(fb1rvi2-fbcrvi2(j,k))*epsi
         b1bc(4,m,3,j,k)=(fb1rwi2-fbcrwi2(j,k))*epsi
         b1bc(5,m,3,j,k)=(fb1ei2-fbcei2(j,k))*epsi
      else
         row(1) = ltog(ijk+1)
         row(2) = row(1) + 1
         row(3) = row(2) + 1
         row(4) = row(3) + 1
         row(5) = row(4) + 1
         col(2) = ltog(ijk+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
c         bw(1)=(fb1ri2-fbcri2(j,k))*epsi
c         bw(2)=(fb1rui2-fbcrui2(j,k))*epsi
c         bw(3)=(fb1rvi2-fbcrvi2(j,k))*epsi
c         bw(4)=(fb1rwi2-fbcrwi2(j,k))*epsi
c         bw(5)=(fb1ei2-fbcei2(j,k))*epsi
         bw(1)=(fb1ri2-fbcri2(j,k))*epsi
         bw(3)=(fb1rui2-fbcrui2(j,k))*epsi
         bw(5)=(fb1rvi2-fbcrvi2(j,k))*epsi
         bw(7)=(fb1rwi2-fbcrwi2(j,k))*epsi
         bw(9)=(fb1ei2-fbcei2(j,k))*epsi
         col(1) = ltog(ijkn+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
         call MatSetValues(mat,5,row,2,col,bw,IV,ierr)
      endif
    8 CONTINUE
      endif
c
      if (xsf1 .eq. 1) then
        i=1
        do 9 k=zsf2,zef01
           ik = i-gxsf + (k-gzsf)*gxm*gym
        do 9 j=ysf2,yef01
           ijk  = ndof * (ik + (j-gysf)*gxm) - 1
           ijkn = ijk + ndof
      r1=R(1,J,K)+qel(1)
      ru1=RU(1,J,K)+qel(2)
      rv1=RV(1,J,K)+qel(3)
      rw1=RW(1,J,K)+qel(4)
      e1=E(1,J,K)+qel(5)
      r2=R(2,J,K)
      ru2=RU(2,J,K)
      rv2=RV(2,J,K)
      rw2=RW(2,J,K)
      e2=E(2,J,K)
      p1=gm1*(e1-p5*(ru1**2+rv1**2+rw1**2)
     * /r1)
      p2=gm1*(e2-p5*(ru2**2+rv2**2+rw2**2)
     * /r2)
      R0=p5*(R1+R2)
      C0=SQRT(GAM*PINIT/R0)
      R1JK=R2+2.*(PINIT-P2)/C0**2
      fb1ri1=R1-R2-2.*(PINIT-P2)/C0**2
      CON=two*(PINIT-P2)/(SADAI(1,J,K)*R0*C0)
      U=RU2/R2+AIX(1,J,K)*CON
      V=RV2/R2+AIY(1,J,K)*CON
      W=RW2/R2+AIZ(1,J,K)*CON
      fb1rui1=RU1-U*R1JK
      fb1rvi1=RV1-V*R1JK
      fb1rwi1=RW1-W*R1JK
      fb1ei1=E1-PINIT*RGM1-p5*R1JK*(U**2+V**2+W**2)
c     Diagonal term (i=1)
      if (nloc .eq. -1) then
         b1bc(1,m,1,j,k)=(fb1ri1-fbcri1(j,k))*epsi
         b1bc(2,m,1,j,k)=(fb1rui1-fbcrui1(j,k))*epsi
         b1bc(3,m,1,j,k)=(fb1rvi1-fbcrvi1(j,k))*epsi
         b1bc(4,m,1,j,k)=(fb1rwi1-fbcrwi1(j,k))*epsi
         b1bc(5,m,1,j,k)=(fb1ei1-fbcei1(j,k))*epsi
      else
c         bw(1)=(fb1ri1-fbcri1(j,k))*epsi
c         bw(2)=(fb1rui1-fbcrui1(j,k))*epsi
c         bw(3)=(fb1rvi1-fbcrvi1(j,k))*epsi
c         bw(4)=(fb1rwi1-fbcrwi1(j,k))*epsi
c         bw(5)=(fb1ei1-fbcei1(j,k))*epsi
         bw(1)=(fb1ri1-fbcri1(j,k))*epsi
         bw(3)=(fb1rui1-fbcrui1(j,k))*epsi
         bw(5)=(fb1rvi1-fbcrvi1(j,k))*epsi
         bw(7)=(fb1rwi1-fbcrwi1(j,k))*epsi
         bw(9)=(fb1ei1-fbcei1(j,k))*epsi
      endif
c
      r1=R(1,J,K)
      ru1=RU(1,J,K)
      rv1=RV(1,J,K)
      rw1=RW(1,J,K)
      e1=E(1,J,K)
      r2=R(2,J,K)+qel(1)
      ru2=RU(2,J,K)+qel(2)
      rv2=RV(2,J,K)+qel(3)
      rw2=RW(2,J,K)+qel(4)
      e2=E(2,J,K)+qel(5)
      p1=gm1*(e1-p5*(ru1**2+rv1**2+rw1**2)
     * /r1)
      p2=gm1*(e2-p5*(ru2**2+rv2**2+rw2**2)
     * /r2)
      R0=p5*(R1+R2)
      C0=SQRT(GAM*PINIT/R0)
      R1JK=R2+2.*(PINIT-P2)/C0**2
      fb1ri1=R1-R2-2.*(PINIT-P2)/C0**2
      CON=two*(PINIT-P2)/(SADAI(1,J,K)*R0*C0)
      U=RU2/R2+AIX(1,J,K)*CON
      V=RV2/R2+AIY(1,J,K)*CON
      W=RW2/R2+AIZ(1,J,K)*CON
      fb1rui1=RU1-U*R1JK
      fb1rvi1=RV1-V*R1JK
      fb1rwi1=RW1-W*R1JK
      fb1ei1=E1-PINIT*RGM1-p5*R1JK*(U**2+V**2+W**2)
c     Eastern neighbor (i=2)
      if (nloc .eq. -1) then
         b1bc(1,m,2,j,k)=(fb1ri1-fbcri1(j,k))*epsi
         b1bc(2,m,2,j,k)=(fb1rui1-fbcrui1(j,k))*epsi
         b1bc(3,m,2,j,k)=(fb1rvi1-fbcrvi1(j,k))*epsi
         b1bc(4,m,2,j,k)=(fb1rwi1-fbcrwi1(j,k))*epsi
         b1bc(5,m,2,j,k)=(fb1ei1-fbcei1(j,k))*epsi
      else
         row(1) = ltog(ijk+1)
         row(2) = row(1) + 1
         row(3) = row(2) + 1
         row(4) = row(3) + 1
         row(5) = row(4) + 1
         col(1) = ltog(ijk+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
         bw(2)=(fb1ri1-fbcri1(j,k))*epsi
         bw(4)=(fb1rui1-fbcrui1(j,k))*epsi
         bw(6)=(fb1rvi1-fbcrvi1(j,k))*epsi
         bw(8)=(fb1rwi1-fbcrwi1(j,k))*epsi
         bw(10)=(fb1ei1-fbcei1(j,k))*epsi
c         bw(1)=(fb1ri1-fbcri1(j,k))*epsi
c         bw(2)=(fb1rui1-fbcrui1(j,k))*epsi
c         bw(3)=(fb1rvi1-fbcrvi1(j,k))*epsi
c         bw(4)=(fb1rwi1-fbcrwi1(j,k))*epsi
c         bw(5)=(fb1ei1-fbcei1(j,k))*epsi
c         col(1) = ltog(ijkn+m)
         col(2) = ltog(ijkn+m)
c         call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
         call MatSetValues(mat,5,row,2,col,bw,IV,ierr)
      endif

    9 CONTINUE
      endif

      RETURN
      END

c ----------------------------------------------------------------

      integer function nd2 (m,epsi,qer,qel,pass,mat,ltog,nloc,d,
     &               xx,p,
     &               br,bl,be,sadai,sadaj,sadak,
     &               aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &               f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
c
c nd2 - Uses finite differencing to compute a numerical Jacobian
c       estimation for the interior grid points only (no boundary
c       points).  This version directly assembles the PETSc matrix 
c       via calls to MatSetValues().
c
c      This routine is called by jform2() (explicit boundary conditions)
c      and jformdt2() (implicit boundary conditions).
c
#include "param.h"
#include "varbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "sada.h"
#include "freq.h"
#include "es.h"
#include "condi.h"
#include "sigma.h"
#include "diag.h"
#include "fv.h"

C   NDL COMPUTES THE LEFT NUMERICAL JACOBIAN.
c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
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
c
      double precision  BW(5)
      double precision  dq(5,d_ni),f(5,d_ni)
      double precision  ql(5,d_ni),qer(5),qel(5)
      double precision  epsi,tl,pl
      integer i,j,k,l,m,ib,jb,kb,IV,row(5),pass,ijm,nloc
      integer ltog(0:nloc-1),ij,ndof,ierr,col,ijp
      integer ip1,jp1,kp1
      Mat     mat

      nd2 = 0
      if (pass .ne. 1 .and. pass .ne. 2) then
        if (rank .eq. 0) 
     &    write(6,*) 'nd2: invalid value of pass!'
        nd2 = -1
        return
      endif

c     IV corresponds to INSERT_VALUES mode for MatSetValues
      IV = INSERT_VALUES

C   NUMERICAL JACOBIAN IN THE I DIRECTION
      ib = 1
      jb = 2
      kb = 2
      CALL RLVECS(QER,QEL,1,0,0,ib,jb,kb,AIX,AIY,AIZ,SADAI,
     &                   xx,p,br,bl,be)

      do 2 k=gzsf2w,zef01
      do 2 j=gysf2w,yef01
      do 21 i=gxsf1,xef01

c      DO 2 K=2,NK
c      DO 2 J=2,NJ
c      DO 21 I=1,NI
      ql(1,I)=R(I,J,K)+qel(1)
      ql(2,I)=RU(I,J,K)+qel(2)
      ql(3,I)=RV(I,J,K)+qel(3)
      ql(4,I)=RW(I,J,K)+qel(4)
      ql(5,I)=E(I,J,K)+qel(5)
      DQ(1,I)=R(I+1,J,K)+qer(1)-ql(1,i)
      DQ(2,I)=RU(I+1,J,K)+qer(2)-ql(2,i)
      DQ(3,I)=RV(I+1,J,K)+qer(3)-ql(3,i)
      DQ(4,I)=RW(I+1,J,K)+qer(4)-ql(4,i)
      DQ(5,I)=E(I+1,J,K)+qer(5)-ql(5,i)
      DO 21 L=1,5
   21 SP(L,I)=BL(L,1,I,J,K)*DQ(1,I)+BL(L,2,I,J,K)*DQ(2,I)+
     *        BL(L,3,I,J,K)*DQ(3,I)+BL(L,4,I,J,K)*DQ(4,I)+
     *        BL(L,5,I,J,K)*DQ(5,I)

      DO 22 I=gxsf1w,xef01
c      DO 22 I=1,NI
      DO 22 L=1,5
      SM2(L,I)=BE(2,L,I,J,K)*SP(L,I)
   22 SP2(L,I)=BE(1,L,I,J,K)*SP(L,I)
      DO 23 I=gxsf1w,xef01
c      DO 23 I=1,NI
      DO 23 L=1,5
   23 F(L,I)=BR(L,1,I,J,K)*SM2(1,I)+BR(L,2,I,J,K)*SM2(2,I)+
     *       BR(L,3,I,J,K)*SM2(3,I)+BR(L,4,I,J,K)*SM2(4,I)+
     *       BR(L,5,I,J,K)*SM2(5,I)

      DO 2 I=gxsf1w,xef01
      ij  = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &           (k-gzsf)*gxm*gym) - 1
      ijm = ij - ndof
      ijp = ij + ndof
      ip1 = i + 1
      TL=(ql(2,i)*AIX(I,J,K)+ql(3,i)*AIY(I,J,K)+
     *    ql(4,i)*AIZ(I,J,K))/ql(1,i)
      pl=gm1*(ql(5,i)-p5*(ql(2,i)**2+ql(3,i)**2+ql(4,i)**2)
     * /ql(1,i))
      F(1,i)=ql(1,i)*TL+F(1,i)
      F(2,i)=ql(2,i)*TL+AIX(I,J,K)*pl+F(2,i)
      F(3,i)=ql(3,i)*TL+AIY(I,J,K)*pl+F(3,i)
      F(4,i)=ql(4,i)*TL+AIZ(I,J,K)*pl+F(4,i)
      F(5,i)=(ql(5,i)+pl)*TL+F(5,i)
c
c  Although we loop over ghost points in order to set arrays
c  such as sm2, sp2, sp, we only set matrix data for the 
c  locally owned rows of each processor.
c
      if (pass .eq. 1) then
c        corresponds to b1() - west neighbor
         bw(1) = (f1(1,i,j,k)-f(1,i))*epsi
         bw(2) = (f1(2,i,j,k)-f(2,i))*epsi
         bw(3) = (f1(3,i,j,k)-f(3,i))*epsi
         bw(4) = (f1(4,i,j,k)-f(4,i))*epsi
         bw(5) = (f1(5,i,j,k)-f(5,i))*epsi
         if (i .lt. ni) then
         if (i+1 .ge. xsf1 .and. i+1 .le. xef01 .and. 
     &       j .ge. ysf1 .and. j .le. yef01 .and.
     &       k .ge. zsf1 .and. k .le. zef01) then
            if (bctype .eq. IMPLICIT .or. i .gt. 1) then
               row(1) = ltog(ijp+1)
               row(2) = row(1)+1
               row(3) = row(1)+2
               row(4) = row(1)+3
               row(5) = row(1)+4
               col    = ltog(ij+m)
               call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
            endif
         endif
         endif
c         if (i .gt. 1) then      
         if (i .ge. xsf2 .and. i .le. xef01 .and. 
     &       j .ge. ysf2 .and. j .le. yef01 .and.
     &       k .ge. zsf2 .and. k .le. zef01) then
            d(1,m,i,j,k) = d(1,m,i,j,k) - bw(1)
            d(2,m,i,j,k) = d(2,m,i,j,k) - bw(2)
            d(3,m,i,j,k) = d(3,m,i,j,k) - bw(3)
            d(4,m,i,j,k) = d(4,m,i,j,k) - bw(4)
            d(5,m,i,j,k) = d(5,m,i,j,k) - bw(5)
         endif
      else 
c        corresponds to b4() - east neighbor
         bw(1) = (f(1,i)-f1(1,i,j,k))*epsi
         bw(2) = (f(2,i)-f1(2,i,j,k))*epsi
         bw(3) = (f(3,i)-f1(3,i,j,k))*epsi
         bw(4) = (f(4,i)-f1(4,i,j,k))*epsi
         bw(5) = (f(5,i)-f1(5,i,j,k))*epsi
         if (i .gt. 1) then
         if (i .ge. xsf1 .and. i .le. xef01 .and. 
     &       j .ge. ysf1 .and. j .le. yef01 .and.
     &       k .ge. zsf1 .and. k .le. zef01) then
            if (bctype .eq. IMPLICIT .or. i .lt. ni) then
               row(1) = ltog(ij+1)
               row(2) = row(1)+1
               row(3) = row(1)+2
               row(4) = row(1)+3
               row(5) = row(1)+4
               col    = ltog(ijp+m)
               call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
            endif
         endif
         endif
c         if (i .lt. ni) then
         if (ip1 .ge. xsf2 .and. ip1 .le. xef01 .and. 
     &       j .ge. ysf2 .and. j .le. yef01 .and.
     &       k .ge. zsf2 .and. k .le. zef01) then
            d(1,m,ip1,j,k) = d(1,m,ip1,j,k) - bw(1)
            d(2,m,ip1,j,k) = d(2,m,ip1,j,k) - bw(2)
            d(3,m,ip1,j,k) = d(3,m,ip1,j,k) - bw(3)
            d(4,m,ip1,j,k) = d(4,m,ip1,j,k) - bw(4)
            d(5,m,ip1,j,k) = d(5,m,ip1,j,k) - bw(5)
         endif
      endif
    2 continue

C   NUMERICAL JACOBIAN IN THE J DIRECTION
      ib = 2
      jb = 1
      kb = 2
      CALL RLVECS(QER,QEL,0,1,0,ib,jb,kb,AJX,AJY,AJZ,SADAJ,
     &                   xx,p,br,bl,be)

      do 3 k=gzsf2w,zef01
      do 3 i=gxsf2w,xef01
      do 31 j=gysf1,yef01
c      DO 3 K=2,NK
c      DO 3 I=2,NI
c      DO 31 J=1,NJ
      ql(1,J)=R(I,J,K)+qel(1)
      ql(2,J)=RU(I,J,K)+qel(2)
      ql(3,J)=RV(I,J,K)+qel(3)
      ql(4,J)=RW(I,J,K)+qel(4)
      ql(5,J)=E(I,J,K)+qel(5)
      DQ(1,J)=R(I,J+1,K)+qer(1)-ql(1,j)
      DQ(2,J)=RU(I,J+1,K)+qer(2)-ql(2,j)
      DQ(3,J)=RV(I,J+1,K)+qer(3)-ql(3,j)
      DQ(4,J)=RW(I,J+1,K)+qer(4)-ql(4,j)
      DQ(5,J)=E(I,J+1,K)+qer(5)-ql(5,j)
      DO 31 L=1,5
   31 SP(L,J)=BL(L,1,I,J,K)*DQ(1,J)+BL(L,2,I,J,K)*DQ(2,J)+
     *        BL(L,3,I,J,K)*DQ(3,J)+BL(L,4,I,J,K)*DQ(4,J)+
     *        BL(L,5,I,J,K)*DQ(5,J)
      DO 32 J=gysf1w,yef01
c      DO 32 J=1,NJ
      DO 32 L=1,5
      SM2(L,J)=BE(2,L,I,J,K)*SP(L,J)
   32 SP2(L,J)=BE(1,L,I,J,K)*SP(L,J)
      DO 33 J=gysf1w,yef01
c      DO 33 J=1,NJ
      DO 33 L=1,5
   33 F(L,J)=BR(L,1,I,J,K)*SM2(1,J)+BR(L,2,I,J,K)*SM2(2,J)+
     *       BR(L,3,I,J,K)*SM2(3,J)+BR(L,4,I,J,K)*SM2(4,J)+
     *       BR(L,5,I,J,K)*SM2(5,J)
      DO 3 J=gysf1w,yef01
         ij  = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &              (k-gzsf)*gxm*gym) - 1
         ijm = ij - ndof*gxm
         ijp = ij + ndof*gxm
         jp1 = j + 1

      TL=(ql(2,j)*AJX(I,J,K)+ql(3,j)*AJY(I,J,K)+
     *    ql(4,j)*AJZ(I,J,K))/ql(1,j)
      pl=gm1*(ql(5,j)-p5*(ql(2,j)**2+ql(3,j)**2+ql(4,j)**2)
     * /ql(1,j))
      F(1,j)=ql(1,j)*TL+F(1,j)
      F(2,j)=ql(2,j)*TL+AJX(I,J,K)*pl+F(2,j)
      F(3,j)=ql(3,j)*TL+AJY(I,J,K)*pl+F(3,j)
      F(4,j)=ql(4,j)*TL+AJZ(I,J,K)*pl+F(4,j)
      F(5,j)=(ql(5,j)+pl)*TL+F(5,j)
c
c  Although we loop over ghost points in order to set arrays
c  such as sm2, sp2, sp, we only set matrix data for the
c  locally owned rows of each processor.
c
      if (pass .eq. 1) then
c        corresponds to b2() - south neighbor
         bw(1) = (g1(1,i,j,k)-f(1,j))*epsi
         bw(2) = (g1(2,i,j,k)-f(2,j))*epsi
         bw(3) = (g1(3,i,j,k)-f(3,j))*epsi
         bw(4) = (g1(4,i,j,k)-f(4,j))*epsi
         bw(5) = (g1(5,i,j,k)-f(5,j))*epsi
         if (j .lt. nj) then
         if (i .ge. xsf1 .and. i .le. xef01 .and. 
     &       j+1 .ge. ysf1 .and. j+1 .le. yef01 .and.
     &       k .ge. zsf1 .and. k .le. zef01) then
            if (bctype .eq. IMPLICIT .or. j .gt. 1) then
               row(1) = ltog(ijp+1)
               row(2) = row(1)+1
               row(3) = row(1)+2
               row(4) = row(1)+3
               row(5) = row(1)+4
               col    = ltog(ij+m)
               call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
            endif
         endif
         endif
c         if (j .gt. 1) then
         if (i .ge. xsf2 .and. i .le. xef01 .and. 
     &       j .ge. ysf2 .and. j .le. yef01 .and.
     &       k .ge. zsf2 .and. k .le. zef01) then
            d(1,m,i,j,k) = d(1,m,i,j,k) - bw(1)
            d(2,m,i,j,k) = d(2,m,i,j,k) - bw(2)
            d(3,m,i,j,k) = d(3,m,i,j,k) - bw(3)
            d(4,m,i,j,k) = d(4,m,i,j,k) - bw(4)
            d(5,m,i,j,k) = d(5,m,i,j,k) - bw(5)
         endif
      else
c        corresponds to b5() - north neighbor
         bw(1) = (f(1,j)-g1(1,i,j,k))*epsi
         bw(2) = (f(2,j)-g1(2,i,j,k))*epsi
         bw(3) = (f(3,j)-g1(3,i,j,k))*epsi
         bw(4) = (f(4,j)-g1(4,i,j,k))*epsi
         bw(5) = (f(5,j)-g1(5,i,j,k))*epsi
         if (j .gt. 1) then
         if (i .ge. xsf1 .and. i .le. xef01 .and. 
     &       j .ge. ysf1 .and. j .le. yef01 .and.
     &       k .ge. zsf1 .and. k .le. zef01) then
            if (bctype .eq. IMPLICIT .or. j .lt. nj) then
               row(1) = ltog(ij+1)
               row(2) = row(1)+1
               row(3) = row(1)+2
               row(4) = row(1)+3
               row(5) = row(1)+4
               col    = ltog(ijp+m)
               call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
            endif
         endif
         endif
c         if (j .lt. nj) then
         if (i .ge. xsf2 .and. i .le. xef01 .and. 
     &       jp1 .ge. ysf2 .and. jp1 .le. yef01 .and.
     &       k .ge. zsf2 .and. k .le. zef01) then
            d(1,m,i,jp1,k) = d(1,m,i,jp1,k) - bw(1)
            d(2,m,i,jp1,k) = d(2,m,i,jp1,k) - bw(2)
            d(3,m,i,jp1,k) = d(3,m,i,jp1,k) - bw(3)
            d(4,m,i,jp1,k) = d(4,m,i,jp1,k) - bw(4)
            d(5,m,i,jp1,k) = d(5,m,i,jp1,k) - bw(5)
         endif
      endif
    3 continue
C   NUMERICAL JACOBIAN IN THE K DIRECTION
      ib = 2
      jb = 2
      kb = 1
      CALL RLVECS(QER,QEL,0,0,1,ib,jb,kb,AKX,AKY,AKZ,SADAK,
     &                   xx,p,br,bl,be)

      do 4 j=gysf2w,yef01
      do 4 i=gxsf2w,xef01
      do 41 k=gzsf1,zef01
c      DO 4 J=2,NJ
c      DO 4 I=2,NI
c      DO 41 K=1,NK
      ql(1,K)=R(I,J,K)+qel(1)
      ql(2,K)=RU(I,J,K)+qel(2)
      ql(3,K)=RV(I,J,K)+qel(3)
      ql(4,K)=RW(I,J,K)+qel(4)
      ql(5,K)=E(I,J,K)+qel(5)
      DQ(1,K)=R(I,J,K+1)+qer(1)-ql(1,k)
      DQ(2,K)=RU(I,J,K+1)+qer(2)-ql(2,k)
      DQ(3,K)=RV(I,J,K+1)+qer(3)-ql(3,k)
      DQ(4,K)=RW(I,J,K+1)+qer(4)-ql(4,k)
      DQ(5,K)=E(I,J,K+1)+qer(5)-ql(5,k)
      DO 41 L=1,5
   41 SP(L,K)=BL(L,1,I,J,K)*DQ(1,K)+BL(L,2,I,J,K)*DQ(2,K)+
     *        BL(L,3,I,J,K)*DQ(3,K)+BL(L,4,I,J,K)*DQ(4,K)+
     *        BL(L,5,I,J,K)*DQ(5,K)
      do 42 k=gzsf1w,zef01
c      DO 42 K=1,NK
      DO 42 L=1,5
      SM2(L,K)=BE(2,L,I,J,K)*SP(L,K)
   42 SP2(L,K)=BE(1,L,I,J,K)*SP(L,K)
      do 43 k=gzsf1w,zef01
c      DO 43 K=1,NK
      DO 43 L=1,5
   43 F(L,K)=BR(L,1,I,J,K)*SM2(1,K)+BR(L,2,I,J,K)*SM2(2,K)+
     *       BR(L,3,I,J,K)*SM2(3,K)+BR(L,4,I,J,K)*SM2(4,K)+
     *       BR(L,5,I,J,K)*SM2(5,K)
      do 4 k=gzsf1w,zef01
c      DO 4 K=1,NK
      ij  = ndof * ((i-gxsf) + (j-gysf)*gxm + 
     &           (k-gzsf)*gxm*gym) - 1
      ijm = ij - ndof*gxm*gym
      ijp = ij + ndof*gxm*gym
      kp1 = k + 1
      TL=(ql(2,k)*AKX(I,J,K)+ql(3,k)*AKY(I,J,K)+
     *    ql(4,k)*AKZ(I,J,K))/ql(1,k)
      pl=gm1*(ql(5,k)-p5*(ql(2,k)**2+ql(3,k)**2+ql(4,k)**2)
     * /ql(1,k))
      F(1,k)=ql(1,k)*TL+F(1,k)
      F(2,k)=ql(2,k)*TL+AKX(I,J,K)*pl+F(2,k)
      F(3,k)=ql(3,k)*TL+AKY(I,J,K)*pl+F(3,k)
      F(4,k)=ql(4,k)*TL+AKZ(I,J,K)*pl+F(4,k)
      F(5,k)=(ql(5,k)+pl)*TL+F(5,k)

c
c  Although we loop over ghost points in order to set arrays
c  such as sm2, sp2, sp, we only set matrix data for the
c  locally owned rows of each processor.
c
      if (pass .eq. 1) then
c        corresponds to array b3() - lower neighbor
         bw(1) = (h1(1,i,j,k)-f(1,k))*epsi
         bw(2) = (h1(2,i,j,k)-f(2,k))*epsi
         bw(3) = (h1(3,i,j,k)-f(3,k))*epsi
         bw(4) = (h1(4,i,j,k)-f(4,k))*epsi
         bw(5) = (h1(5,i,j,k)-f(5,k))*epsi
         if (k .lt. nk) then
         if (i .ge. xsf1 .and. i .le. xef01 .and. 
     &       j .ge. ysf1 .and. j .le. yef01 .and.
     &       k+1 .ge. zsf1 .and. k+1 .le. zef01) then
            if (bctype .eq. IMPLICIT .or. k .gt. 1) then
               row(1) = ltog(ijp+1)
               row(2) = row(1)+1
               row(3) = row(1)+2
               row(4) = row(1)+3
               row(5) = row(1)+4
               col    = ltog(ij+m)
               call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
            endif
         endif
         endif
c         if (k .gt. 1) then      
         if (i .ge. xsf2 .and. i .le. xef01 .and. 
     &       j .ge. ysf2 .and. j .le. yef01 .and.
     &       k .ge. zsf2 .and. k .le. zef01) then
            d(1,m,i,j,k) = d(1,m,i,j,k) - bw(1)
            d(2,m,i,j,k) = d(2,m,i,j,k) - bw(2)
            d(3,m,i,j,k) = d(3,m,i,j,k) - bw(3)
            d(4,m,i,j,k) = d(4,m,i,j,k) - bw(4)
            d(5,m,i,j,k) = d(5,m,i,j,k) - bw(5)
         endif
      else
c        corresponds to array b6() - upper neighbor
         bw(1) = (f(1,k)-h1(1,i,j,k))*epsi
         bw(2) = (f(2,k)-h1(2,i,j,k))*epsi
         bw(3) = (f(3,k)-h1(3,i,j,k))*epsi
         bw(4) = (f(4,k)-h1(4,i,j,k))*epsi
         bw(5) = (f(5,k)-h1(5,i,j,k))*epsi
         if (k .gt. 1) then
         if (i .ge. xsf1 .and. i .le. xef01 .and. 
     &       j .ge. ysf1 .and. j .le. yef01 .and.
     &       k .ge. zsf1 .and. k .le. zef01) then
            if (bctype .eq. IMPLICIT .or. k .lt. nk) then
               row(1) = ltog(ij+1)
               row(2) = row(1)+1
               row(3) = row(1)+2
               row(4) = row(1)+3
               row(5) = row(1)+4
               col    = ltog(ijp+m)
               call MatSetValues(mat,5,row,1,col,bw,IV,ierr)
            endif
         endif
         endif
c         if (k .lt. nk) then
         if (i .ge. xsf2 .and. i .le. xef01 .and. 
     &       j .ge. ysf2 .and. j .le. yef01 .and.
     &       kp1 .ge. zsf2 .and. kp1 .le. zef01) then
            d(1,m,i,j,kp1) = d(1,m,i,j,kp1) - bw(1)
            d(2,m,i,j,kp1) = d(2,m,i,j,kp1) - bw(2)
            d(3,m,i,j,kp1) = d(3,m,i,j,kp1) - bw(3)
            d(4,m,i,j,kp1) = d(4,m,i,j,kp1) - bw(4)
            d(5,m,i,j,kp1) = d(5,m,i,j,kp1) - bw(5)
         endif
      endif

    4 continue
      return
      end

