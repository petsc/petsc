c
c  This file contains routines that are used only within the PETSc
c  solver.  Most are interface routines to pass between the PETSc
c  data structures, solvers, etc. and the Julianne code.
c
c -------------------------------------------------------------------
c
c  parsetup - Sets local grid info for Fortran modules.
c
c  Note the notational differences between Fortran code and C code:
c    C code:  base grid is nx-1,ny-1,nz-1, ghost points accordingly
c    Fortran code:  need to deal with nx, nx+1, ...
c        C code (xefp2,gxefp2) -> Fortran code (xefp1,gxefp1)
c        Seems most natural for both parts of application,
c          although admittedly somewhat confusing.
c
      integer function parsetup(Acomm, Aprintg, Ano_output,
     &        Abctype, Arank, Asize, Aproblem,
     &        Agxsf, Agysf, Agzsf, Agxef, Agyef, Agzef, 
     &        Axsf, Aysf, Azsf, Axef, Ayef, Azef, 
     &        Axefm1, Ayefm1, Azefm1, Axefp1, Ayefp1, Azefp1, 
     &        Agxefp1, Agyefp1, Agzefp1,
     &        Axsf1, Aysf1, Azsf1, Agxsf1, Agysf1, Agzsf1,
     &        Axsf2, Aysf2, Azsf2, Agxsf2, Agysf2, Agzsf2,
     &        Agxsfw, Agysfw, Agzsfw, Agxefw, Agyefw, Agzefw,
     &        Agxm, Agym, Agzm,
     &        Axef01, Ayef01, Azef01, Agxef01, Agyef01, Agzef01)

#include "param.h"

      integer Aprintg, Acomm, Arank, Asize, Abctype, Aproblem,
     &        Agxsf, Agysf, Agzsf, Agxef, Agyef, Agzef,
     &        Axsf, Aysf, Azsf, Axef, Ayef, Azef, 
     &        Axefm1, Ayefm1, Azefm1, Axefp1, Ayefp1, Azefp1, 
     &        Agxefp1, Agyefp1, Agzefp1, 
     &        Axsf1, Aysf1, Azsf1, Agxsf1, Agysf1, Agzsf1,
     &        Axsf2, Aysf2, Azsf2, Agxsf2, Agysf2, Agzsf2,
     &        Agxsfw, Agysfw, Agzsfw, Agxefw, Agyefw, Agzefw,
     &        Agxm, Agym, Agzm, ierr, Ano_output,
     &        Axef01, Ayef01, Azef01, Agxef01, Agyef01, Agzef01

c  in param.h: Common block, pgrid, for local grid data:

      parsetup  = 0
      comm      = Acomm
      printg    = Aprintg
      bctype    = Abctype
      rank      = Arank
      size      = Asize
      problem   = Aproblem
      no_output = Ano_output

      if (problem .eq. 1) then
         OPEN (2,FILE='m6c',STATUS='OLD')
         ni = 49
         nj = 9
         nk = 9
         if (d_ni .lt. 49  .or. d_nj .lt. 9 .or. d_nk .lt. 9) then
           if (rank .eq. 0) 
     &        write(6,*) 'Bad dimension for problem 1: ni,nj,nk = ',
     &                    ni,nj,nk
     
           parsetup = -1
           return
         endif
      else if (problem .eq. 2) then
         OPEN (2,FILE='m6f',STATUS='OLD')
         ni = 97
         nj = 17
         nk = 17
         if (d_ni .lt. 97  .or. d_nj .lt. 17 .or. d_nk .lt. 17) then
           if (rank .eq. 0)
     &        write(6,*) 'Bad dimension for problem 2: ni,nj,nk = ',
     &                    ni,nj,nk
           parsetup = -1
           return
         endif
      else if (problem .eq. 3) then
         OPEN (2,FILE='m6n',STATUS='OLD')
         ni = 193
         nj = 33
         nk = 33
         if (d_ni .lt. 193  .or. d_nj .lt. 33 .or. d_nk .lt. 33) then
           if (rank .eq. 0)
     &        write(6,*) 'Bad dimension for problem 3: ni,nj,nk = ',
     &                    ni,nj,nk
           parsetup = -1
           return
         endif
      else
         if (rank .eq. 0) write(6,*) 'Only probs 1, 2, 3 are supported'
         parsetup = -1
         return
      endif
      ni1 = ni+1
      nj1 = nj+1
      nk1 = nk+1

      gxsf    = Agxsf
      gysf    = Agysf
      gzsf    = Agzsf
      gxef    = Agxef
      gyef    = Agyef
      gzef    = Agzef
      gxef01  = Agxef01
      gyef01  = Agyef01
      gzef01  = Agzef01

      xsf     = Axsf
      ysf     = Aysf
      zsf     = Azsf
      xsf1    = Axsf1
      ysf1    = Aysf1
      zsf1    = Azsf1

      xef     = Axef
      yef     = Ayef
      zef     = Azef
      xefm1   = Axefm1
      yefm1   = Ayefm1
      zefm1   = Azefm1
      xef01   = Axef01
      yef01   = Ayef01
      zef01   = Azef01
      xefp1   = Axefp1
      yefp1   = Ayefp1
      zefp1   = Azefp1

      xm      = xef - xsf + 1
      ym      = yef - ysf + 1
      zm      = zef - zsf + 1

      gxsf1   = Agxsf1
      gysf1   = Agysf1
      gzsf1   = Agzsf1

      gxefp1  = Agxefp1
      gyefp1  = Agyefp1
      gzefp1  = Agzefp1

      gxm     = Agxm
      gym     = Agym
      gzm     = Agzm

      xsf2    = Axsf2
      ysf2    = Aysf2
      zsf2    = Azsf2

      gxsf2   = Agxsf2
      gysf2   = Agysf2
      gzsf2   = Agzsf2

      gxsfw   = Agxsfw
      gysfw   = Agysfw
      gzsfw   = Agzsfw

      gxefw   = Agxefw
      gyefw   = Agyefw
      gzefw   = Agzefw

      if (gxsf2 .eq. 2) then
        gxsf2w = 2
        gxsf1w = 1
      else
        gxsf2w = Agxsfw
        gxsf1w = Agxsfw
      endif
      if (gysf2 .eq. 2) then
        gysf2w = 2
        gysf1w = 1
      else 
        gysf2w = Agysfw
        gysf1w = Agysfw
      endif
      if (gzsf2 .eq. 2) then
        gzsf2w = 2
        gzsf1w = 1
      else
        gzsf2w = Agzsfw
        gzsf1w = Agzsfw
      endif

      if (printg .eq. 1) then
      if (rank .eq. 0) write(6,*) 'Problem is ',problem
      call PetscSequentialPhaseBegin(comm,1,ierr)
      write(6,110) rank,xsf,xsf1,xsf2,xefm1,xef01,xefp1,xm
      write(6,120) rank,ysf,ysf1,ysf2,yefm1,yef01,yefp1,ym
      write(6,130) rank,zsf,zsf1,zsf2,zefm1,zef01,zefp1,zm
 110  format('Fort grid: rank ',i4,
     & ': xsf,xsf1,xsf2,xefm1,xef01,xefp1,xm:',7(i4))
 120  format('Fort grid: rank ',i4,
     & ': ysf,ysf1,ysf2,yefm1,yef01,yefp1,ym:',7(i4))
 130  format('Fort grid: rank ',i4,
     & ': zsf,zsf1,zsf2,zefm1,zef01,zefp1,zm:',7(i4))
      write(6,140) rank,gxsf,gxsf1,gxsf2,gxef01,gxefp1,gxm
      write(6,150) rank,gysf,gysf1,gysf2,gyef01,gyefp1,gym
      write(6,160) rank,gzsf,gzsf1,gzsf2,gzef01,gzefp1,gzm
 140  format('Fort ghosts: rank ',i4,
     & ': gxsf,gxsf1,gxsf2,gxef01,gxefp1,gxm:',8(i4))
 150  format('Fort ghosts: rank ',i4,
     & ': gysf,gysf1,gysf2,gyef01,gyefp1,gym:',8(i4))
 160  format('Fort ghosts: rank ',i4,
     & ': gzsf,gzsf1,gzsf2,gzef01,gzefp1,gzm:',8(i4))
      write(6,170) rank,gxsfw,gxsf1w,gxsf2w,gxefw
      write(6,180) rank,gysfw,gysf1w,gysf2w,gyefw
      write(6,190) rank,gzsfw,gzsf1w,gzsf2w,gzefw
 170  format('Fort ghosts (interior): rank ',
     & i4,': gxsfw,gxsf1w,gxsf2w,gxefw:',4(i4))
 180  format('Fort ghosts (interior): rank ',
     & i4,': gysfw,gysf1w,gysf2w,gyefw:',4(i4))
 190  format('Fort ghosts (interior): rank ',
     & i4,': gzsfw,gzsf1w,gzsf2w,gzefw:',4(i4))
#if !defined(PARCH_rs6000)
      call flush(6)
#endif
      call PetscSequentialPhaseEnd(comm,1,ierr)
      endif

      return
      end

c -------------------------------------------------------------------

      integer function jform(iskip,b1,b2,b3,b4,b5,b6,d,dt,
     &                       r,ru,rv,rw,e,p, 
     &                       br,bl,be,sadai,sadaj,sadak,
     &                       aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                       f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
c
c jform - Forms the Jacobian matrix, J(x).  This routine is called by
c         FormJacobian().
c
#include "param.h"
#include "matrixb.h"
#include "diag.h"
#include "dt.h"
#include "varbls.h"
#include "consts.h"
#include "freq.h"
#include "es.h"
#include "sada.h"
#include "avctrs.h"
#include "fv.h"
#include "sigma.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /FREQ/ IFREQ,ISGS
      Double qer(5),qel(5),eps,dql,dqr
      integer          ISKIP,i,j,k,l,m

      jform = 0
      ISKIP = 0
      NCYC=NCYC+1
      IF (NCYC.NE.1.AND.MOD(NCYC,IFREQ).NE.0) THEN
         ISKIP = 1
         GO TO 40
      ENDIF

      do 10 m=1,5
         qel(m)=0.
         qer(m)=0.
 10   continue

      eps=1.0e-07
      do 30 m=1,5
      qel(m)=eps
c Form Jacobian elements in left part of matrix
      call nd (m,eps,qer,qel,b1,b2,b3,r,ru,rv,rw,e,p,
     &         br,bl,be,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &         f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)

      qel(m)=0.
   30 continue
      do 31 m=1,5
      qer(m)=eps
c Form Jacobian elements in right part of matrix
      call nd (m,eps,qer,qel,b4,b5,b6,
     &         r,ru,rv,rw,e,p,
     &         br,bl,be,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &         f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
      qer(m)=0.
   31 continue
c Find eigendecomposition of diagonal block of Jacobian; Form DT matrix.
c But really need to recalculate dt only when the iterates change. 
      call eigenv(dt,r,ru,rv,rw,e,p,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
      DO 21 M=1,5
      DO 21 L=1,5
      do 21 k=zsf2,zef01
      do 21 j=ysf2,yef01
      do 21 i=xsf2,xef01
c      DO 21 K=2,NK
c      DO 21 J=2,NJ
c      DO 21 I=2,NI
      dql=B1(L,M,I,J,K)+B2(L,M,I,J,K)+B3(L,M,I,J,K)
      dqr=B4(L,M,I-1,J,K)+B5(L,M,I,J-1,K)+B6(L,M,I,J,K-1)
      D(L,M,I,J,K)=dql-dqr
      IF (L.EQ.M) D(L,M,I,J,K)=D(L,M,I,J,K)+1.0/DT(I,J,K)
   21 CONTINUE
   40 continue

      return
      end

c -------------------------------------------------------------------

      integer function jform2(iskip,ltog,nloc,mat,d,dt,
     &                       r,ru,rv,rw,e,p, 
     &                       br,bl,be,sadai,sadaj,sadak,
     &                       aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                       f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
c
c jform - Forms the Jacobian matrix, J(x).  This routine is called by
c         FormJacobian().
c
#include "param.h"
#include "dt.h"
#include "varbls.h"
#include "consts.h"
#include "freq.h"
#include "es.h"
#include "sada.h"
#include "avctrs.h"
#include "diag.h"
#include "fv.h"
#include "sigma.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /FREQ/ IFREQ,ISGS
      Double qer(5),qel(5),eps,dti
      integer          ltog(0:nloc-1),nloc
      integer          ISKIP,m,ierr,nd2,ijk,jk,rc(5),nc,i,j,k
      Mat              mat

      jform2 = 0
      ISKIP = 0
      NCYC=NCYC+1
      IF (NCYC.NE.1.AND.MOD(NCYC,IFREQ).NE.0) THEN
         ISKIP = 1
         GO TO 40
      ENDIF
      nc = 5

      do 10 m=1,5
         qel(m)=0.
         qer(m)=0.
 10   continue

      eps=1.0e-07
      do 30 m=1,5
      qel(m)=eps
c Form Jacobian elements in left part of matrix
      ierr = nd2 (m,eps,qer,qel,1,mat,ltog,nloc,d,
     &         r,ru,rv,rw,e,p,
     &         br,bl,be,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &         f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
      if (ierr .ne. 0) then
         jform2 = -1
         return 
      endif
      qel(m)=0.
   30 continue

      do 31 m=1,5
      qer(m)=eps
c Form Jacobian elements in right part of matrix
      ierr = nd2 (m,eps,qer,qel,2,mat,ltog,nloc,d,
     &         r,ru,rv,rw,e,p,
     &         br,bl,be,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &         f1,g1,h1,sp,sm,sp1,sp2,sm1,sm2)
      if (ierr .ne. 0) then
         jform2 = -1
         return 
      endif
      qer(m)=0.
   31 continue

c Find eigendecomposition of diagonal block of Jacobian; Form DT matrix.
c But really need to recalculate dt only when the iterates change. 
      call eigenv(dt,r,ru,rv,rw,e,p,sadai,sadaj,sadak,
     &         aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)

c We must specify that we are inserting matrix blocks by columns, since
c the PETSc default is row-oriented input.
      call MatSetOption(mat,MAT_COLUMN_ORIENTED,ierr)

      do 21 k=zsf2,zef01
      do 21 j=ysf2,yef01
       jk = (j-gysf)*gxm + (k-gzsf)*gxm*gym
      do 21 i=xsf2,xef01
c      dql=B1(L,M,I,J,K)+B2(L,M,I,J,K)+B3(L,M,I,J,K)
c      dqr=B4(L,M,I-1,J,K)+B5(L,M,I,J-1,K)+B6(L,M,I,J,K-1)
c      D(L,M,I,J,K)=dql-dqr
c      IF (L.EQ.M) D(L,M,I,J,K)=D(L,M,I,J,K)+1.0/DT(I,J,K)
       ijk = ltog(nc * (i-gxsf + jk))
       rc(1) = ijk
       rc(2) = ijk + 1
       rc(3) = ijk + 2
       rc(4) = ijk + 3
       rc(5) = ijk + 4
       dti = 1.0/DT(I,J,K)
       D(1,1,I,J,K)=D(1,1,I,J,K) + dti
       D(2,2,I,J,K)=D(2,2,I,J,K) + dti
       D(3,3,I,J,K)=D(3,3,I,J,K) + dti
       D(4,4,I,J,K)=D(4,4,I,J,K) + dti
       D(5,5,I,J,K)=D(5,5,I,J,K) + dti
       call MatSetValues(mat,nc,rc,nc,rc,d(1,1,i,j,k),
     &                   INSERT_VALUES,ierr)
   21 CONTINUE

C  The rest of the input is row-oriented.
      call MatSetOption(mat,MAT_ROW_ORIENTED,ierr)

   40 continue

      return
      end

c -------------------------------------------------------------------

      integer function jpressure(r,ru,rv,rw,e,p)

c jpress - Updates the pressure terms after a Newton step.
c
#include "param.h"
#include "varbls.h"
#include "consts.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
      Double rqdq
      integer          i,j,k

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
      P(I,J,K)=GM1*(E(I,J,K)-0.5*RQDQ)
   15 CONTINUE

      RETURN
      END

c -------------------------------------------------------------------

      integer function jmonitor(fnorm,cfl_new,work,r,ru,rv,rw,e,p,
     &                          dr,dru,drv,drw,de,
     &                          aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz)
c
c jmonitor - Writes miscellaneous output information.
c
#include "param.h"
#include "varbls.h"
#include "dvarbls.h"
#include "consts.h"
#include "wing.h"
#include "avctrs.h"
#include "freq.h"
#include "ang.h"

c      COMMON /CONST1/ GAM,RGAM,GM1,RGM1,GS2,FSMACH,CFL,NCYC
c      COMMON /CONST2/ PINIT
c      COMMON /VARBLS/ R(NI1,NJ1,NK1),RU(NI1,NJ1,NK1),RV(NI1,NJ1,NK1)
c      COMMON /VARBLS/ RW(NI1,NJ1,NK1),E(NI1,NJ1,NK1),P(NI1,NJ1,NK1)
c      COMMON /AVCTRS/ AIX(NI,NJ,NK),AJX(NI,NJ,NK),AKX(NI,NJ,NK)
c      COMMON /AVCTRS/ AIY(NI,NJ,NK),AJY(NI,NJ,NK),AKY(NI,NJ,NK)
c      COMMON /AVCTRS/ AIZ(NI,NJ,NK),AJZ(NI,NJ,NK),AKZ(NI,NJ,NK)
c      COMMON /DVRBLS/ DR(NI1,NJ1,NK1),DRU(NI1,NJ1,NK1),DRV(NI1,NJ1,NK1)
c      COMMON /DVRBLS/ DRW(NI1,NJ1,NK1),DE(NI1,NJ1,NK1)
c      COMMON /ANG/ ALPHA,BETA,PHI
c      COMMON /WING/ ITL,ITU,ILE,KTIP
c      COMMON /FREQ/ IFREQ,ISGS

      common /junkvar/ rtrmsr, etrmsr

      Double drecv(3),dsend(3),fnorm,rtmax,rtrms,etmax,etrms
      Double dsup,con,rt,et,rtrmsr,etrmsr,tcl,tcd,areas,cn,cc
      Double area,pbody,cls,cds,xl2,one,work(*),cfl_new
      integer          i,j,k,ierr,nsup

      jmonitor = 0
      cfl = cfl_new
      xl2=log10(fnorm)
      if (no_output .eq. 0) write(29,*) ncyc, xl2
      IF (NCYC.NE.1.AND.MOD(NCYC,5).NE.0) RETURN
c Compute and tabulate norms and the number of supersonic points in flowfield
      RTMAX=0.
      RTRMS=0.
      ETMAX=0.
      ETRMS=0.
      NSUP=0
      dsup=0.
      one=1.0
      CON=GS2+RGM1
      do 20 k=zsf2,zef01
      do 20 j=ysf2,yef01
      do 20 i=xsf2,xef01
c      DO 20 K=2,NK
c      DO 20 J=2,NJ
c      DO 20 I=2,NI
      IF (E(I,J,K).GT.P(I,J,K)*CON) NSUP=NSUP+1
      RT=DR(I,J,K)
      RTRMS=RTRMS+RT**2
      ET=DE(I,J,K)
      ETRMS=ETRMS+ET**2
      IF (ABS(RT).GT.ABS(RTMAX)) RTMAX=RT
      IF (ABS(ET).GT.ABS(ETMAX)) ETMAX=ET
   20 CONTINUE

      dsend(1) = rtmax
      dsend(2) = etmax
      call MPI_Allgather(dsend,2,MPI_DOUBLE_PRECISION,work,2,
     &                   MPI_DOUBLE_PRECISION,comm,ierr)
      rtmax = work(1)
      etmax = work(2)
      do 30 i=2,size
        if (abs(work(2*i-1)) .gt. abs(rtmax)) rtmax = work(2*i-1)
        if (abs(work(2*i)) .gt. abs(etmax)) etmax = work(2*i)
  30  continue
      dsend(1) = rtrms
      dsend(2) = etrms
      dsend(3) = one*nsup
      call MPI_Allreduce(dsend,drecv,3,MPI_DOUBLE_PRECISION,
     &                   MPI_SUM,comm,ierr)
      rtrms = drecv(1)
      etrms = drecv(2)
      dsup  = drecv(3)
      nsup  = int(dsup)

      IF (NCYC.EQ.1) RTRMSR=RTRMS
      IF (NCYC.EQ.1) ETRMSR=ETRMS
      RTRMS=SQRT(RTRMS/RTRMSR)
      ETRMS=SQRT(ETRMS/ETRMSR)
      RTRMS=LOG10(RTRMS)
      ETRMS=LOG10(ETRMS)
      TCL=0.
      TCD=0.
      AREAS=0.

      if (ysf1 .eq. 1) then
        DO 41 K=zsf2,KTIPP
c        DO 41 K=2,KTIP
          CN=0.
          CC=0.
          AREA=0.
          DO 42 I=ITLP1,ITUP
c          DO 42 I=ITL+1,ITU
            AREA=AREA+ABS(AJY(I,1,K))*.5
            PBODY=0.5*(P(I,1,K)+P(I,2,K))
            CN=CN-(PBODY-RGAM)*AJY(I,1,K)
   42       CC=CC-(PBODY-RGAM)*AJX(I,1,K)
          CLS=CN*COS(-BETA)-CC*SIN(-BETA)
          CDS=CN*SIN(-BETA)+CC*COS(-BETA)
          AREAS=AREAS+AREA
          TCL=TCL+CLS
   41     TCD=TCD+CDS
        dsend(1) = tcl
        dsend(2) = tcd
        dsend(3) = areas
      else
        dsend(1) = 0.0
        dsend(2) = 0.0
        dsend(3) = 0.0
      endif
      call MPI_Allreduce(dsend,drecv,3,MPI_DOUBLE_PRECISION,
     &                   MPI_SUM,comm,ierr)
      tcl   = drecv(1)
      tcd   = drecv(2)
      areas = drecv(3)
      TCL=TCL/AREAS*2./FSMACH**2
      TCD=TCD/AREAS*2./FSMACH**2
 1000 continue

      if (no_output .eq. 0 .and. rank .eq. 0) then
        WRITE (6,200) NCYC,RTMAX,RTRMS,ETRMS,XL2,TCL,TCD,NSUP
      endif
  200 FORMAT(I5,1P6E11.3,I5)
      RETURN
      END

c -------------------------------------------------------------------

      subroutine printjul(r,ru,rv,rw,e,p,base)
c
c printjul - Prints grid points (interior only)
c
#include "param.h"
#include "varbls.h"

      integer i,j,k
      integer unit, base

      if (base .lt. 10) base = 50
      unit = base + rank

      do 20 k=zsf2,zef01
      do 20 j=ysf2,yef01
      do 20 i=xsf2,xef01
            write(unit,100) k,j,i,R(I,J,K),Ru(I,J,K),
     &            rv(i,j,k),rw(i,j,k),e(i,j,k),p(i,j,k)
   20    continue
  100 format(i4,i3,i3,6e14.6)
#if !defined(PARCH_rs6000)
      call flush(unit)
#endif
      return
      end

c -------------------------------------------------------------------

      subroutine printgjul(r,ru,rv,rw,e,p,base)
c
c printjul - Prints grid points (exterior/interior and ghost points)
c
#include "param.h"
#include "varbls.h"

      integer i,j,k
      integer unit, base

      if (base .lt. 10) base = 50
      unit = base + rank

      do 20 k=gzsf1,gzefp1
      do 20 j=gysf1,gyefp1
      do 20 i=gxsf1,gxefp1
            write(unit,100) k,j,i,R(I,J,K),Ru(I,J,K),
     &            rv(i,j,k),rw(i,j,k),e(i,j,k),p(i,j,k)
   20    continue
  100 format(i4,i3,i3,6e14.6)
#if !defined(PARCH_rs6000)
      call flush(unit)
#endif
      return
      end

c -------------------------------------------------------------------

      subroutine printbjul(r,ru,rv,rw,e,p,base)
c
c printjul - Prints grid points (exterior/interior and ghost points)
c
#include "param.h"
#include "varbls.h"

      integer i,j,k
      integer unit, base

      if (base .lt. 10) base = 50
      unit = base + rank

      do 20 k=gzsf1,gzefp1
      do 20 j=gysf1,gyefp1
      do 20 i=gxsf1,gxefp1
            if (i .eq. 1 .or. j .eq. 1 .or. k .eq. 1 .or.
     &          i.eq.ni1 .or. j.eq.nj1 .or. k.eq.nk1)
     &            write(unit,100) k,j,i,R(I,J,K),Ru(I,J,K),
     &            rv(i,j,k),rw(i,j,k),e(i,j,k),p(i,j,k)
   20    continue
  100 format(i4,i3,i3,6e14.6)
#if !defined(PARCH_rs6000)
      call flush(unit)
#endif
      return
      end

c -------------------------------------------------------------------
c
c   rscale - Scales input vector by dt.
c
      integer function rscale( dt, dr, dru, drv, drw, de )

#include "param.h"
#include "dt.h"
#include "dvarbls.h"

      integer i,j,k

      rscale = 0
      do 100 k=zsf2,zef01
      do 100 j=ysf2,yef01
      do 100 i=xsf2,xef01
c      do 100 k=2,nk
c      do 100 j=2,nj
c      do 100 i=2,ni
         dr(i,j,k)  = dt(i,j,k)*dr(i,j,k)
         dru(i,j,k) = dt(i,j,k)*dru(i,j,k)
         drv(i,j,k) = dt(i,j,k)*drv(i,j,k)
         drw(i,j,k) = dt(i,j,k)*drw(i,j,k)
         de(i,j,k)  = dt(i,j,k)*de(i,j,k)
 100  continue

      return
      end

c -------------------------------------------------------------------
c
c   readmesh - Reads mesh and wing parameters
c
c  Use these arrays only temporarily for parallel case.  Each processor
c  reads all input data, and then later copies only its local piece and
c  frees the global mesh.  The input file for a particular mesh size is
c  opened in the routine parsetup().
c
      integer function readmesh(itlA,ituA,ileA,ktipA,x,y,z)

#include "param.h"
#include "wing.h"

      Double X(NI,NJ,NK),Y(NI,NJ,NK),Z(NI,NJ,NK)
      integer          itlA, ituA, ileA, ktipA

      readmesh = 0
      READ (2,200) ITLA,ITUA,ILEA,KTIPA
      READ (2,250) X,Y,Z
      itl  = itlA
      itu  = ituA
      ile  = ileA
      ktip = ktipA
      close(2)
  200 FORMAT (4I3)
  250 FORMAT (8F10.5)

      return
      end
