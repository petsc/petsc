c
c  "$Id: xmod.F,v 1.30 1997/10/16 22:09:21 keyes Exp curfman $";
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
      integer function parsetup(Acomm, Aprintg, Anowake,
     &        Ano_output, Abctype, Arank, Asize, Aproblem,
     &        Agxsf, Agysf, Agzsf, Agxef, Agyef, Agzef, 
     &        Axsf, Aysf, Azsf, Axef, Ayef, Azef, 
     &        Axefm1, Ayefm1, Azefm1, Axefp1, Ayefp1, Azefp1, 
     &        Agxefp1, Agyefp1, Agzefp1, Agxefm1, Agyefm1, Agzefm1,
     &        Axsf1, Aysf1, Azsf1, Agxsf1, Agysf1, Agzsf1,
     &        Axsf2, Aysf2, Azsf2, Agxsf2, Agysf2, Agzsf2,
     &        Agxsfw, Agysfw, Agzsfw, Agxefw, Agyefw, Agzefw,
     &        Agxm, Agym, Agzm,
     &        Axef01, Ayef01, Azef01, Agxef01, Agyef01,
     &        Agzef01, Andof, Aggrid, Abcswitch, Amodel,
     &        Abump, Ank1)

#include "param.h"

      integer Aprintg, Acomm, Arank, Asize, Abctype, Anowake,
     &        Aproblem, Agxsf, Agysf, Agzsf, Agxef, Agyef, Agzef,
     &        Axsf, Aysf, Azsf, Axef, Ayef, Azef, Amodel, Ank1,
     &        Axefm1, Ayefm1, Azefm1, Axefp1, Ayefp1, Azefp1, 
     &        Agxefp1, Agyefp1, Agzefp1, ctot, Abcswitch,
     &        Axsf1, Aysf1, Azsf1, Agxsf1, Agysf1, Agzsf1,
     &        Axsf2, Aysf2, Azsf2, Agxsf2, Agysf2, Agzsf2,
     &        Agxsfw, Agysfw, Agzsfw, Agxefw, Agyefw, Agzefw,
     &        Agxm, Agym, Agzm, ierr, Ano_output, Andof, Aggrid,
     &        Axef01, Ayef01, Azef01, Agxef01, Agyef01, Agzef01,
     &        Agxefm1, Agyefm1, Agzefm1

      double precision Abump

c  in param.h: Common block, pgrid, for local grid data:

      parsetup  = 0
      comm      = Acomm
      printg    = Aprintg
      bctype    = Abctype
      rank      = Arank
      size      = Asize
      problem   = Aproblem
      no_output = Ano_output
      ndof      = Andof
      ndof_e    = 5
      bcswitch  = Abcswitch
      nowake    = Anowake
      model     = Amodel
      bump      = Abump

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
      else if (problem .eq. 5) then
         OPEN (2,FILE='m6c_duct',STATUS='OLD')
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
      else if (problem .eq. 6) then
         OPEN (2,FILE='m6f_duct',STATUS='OLD')
         ni = 97
         nj = 17
         nk = 17
         if (d_ni .lt. 97  .or. d_nj .lt. 17 .or. d_nk .lt. 17) then
           if (rank .eq. 0) 
     &        write(6,*) 'Bad dimension for problem 1: ni,nj,nk = ',
     &                    ni,nj,nk
     
           parsetup = -1
           return
         endif
      else
         if (rank .eq. 0) 
     &   write(6,*) 'Only probs 1,2,3 (M6) and 5,6 (duct) are supported'
         parsetup = -1
         return
      endif
      ni1 = ni+1
      nj1 = nj+1
      nk1 = nk+1

c  Fix for 2-dimensional case
      if (Ank1 .eq. 3) then
         nk1_boundary = 3
         nk_boundary  = 2
      else
         nk1_boundary = nk1
         nk_boundary  = nk
      endif

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

      gxefm1  = Agxefm1
      gyefm1  = Agyefm1
      gzefm1  = Agzefm1

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

      if (Aggrid .eq. 1) then
c       Using global grid     
        cx1 = 1
        cxn = ni
        cy1 = 1
        cyn = nj
        cz1 = 1
        czn = nk
      else
c       Using local grid     
        cx1 = max(gxsf1-1,1)
        cxn = gxef01
        cy1 = max(gysf1-1,1)
        cyn = gyef01
        cz1 = max(gzsf1-1,1)
        czn = gzef01
      endif

      if (rank .eq. 0) then 
        write(6,*) 'Problem is ',problem,', ndof = ',ndof
        if (problem .eq. 5) write(6,*) 'Bump = ',bump
        write(6,*) 'Iteration for BC impermeability switch is',bcswitch
      endif
      if (printg .eq. 1) then
      call PetscSequentialPhaseBegin(comm,1,ierr)
      ctot = (cxn-cx1+1)*(cyn-cy1+1)*(czn-cz1+1)
      write(6,100) rank,cx1,cxn,cy1,cyn,cz1,czn,ctot
      write(6,110) rank,xsf,xsf1,xsf2,xefm1,xef01,xefp1,xm
      write(6,120) rank,ysf,ysf1,ysf2,yefm1,yef01,yefp1,ym
      write(6,130) rank,zsf,zsf1,zsf2,zefm1,zef01,zefp1,zm

 100  format('Local coord ranges: rank ',i4,
     & ': cx1,cxn,cy1,cyn,cz1,czn:',6(i4),', ctot:',i8)
 110  format('Fort grid: rank ',i4,
     & ': xsf,xsf1,xsf2,xefm1,xef01,xefp1,xm:',7(i4))
 120  format('Fort grid: rank ',i4,
     & ': ysf,ysf1,ysf2,yefm1,yef01,yefp1,ym:',7(i4))
 130  format('Fort grid: rank ',i4,
     & ': zsf,zsf1,zsf2,zefm1,zef01,zefp1,zm:',7(i4))
      write(6,140) rank,gxsf,gxsf1,gxsf2,gxefm1,gxef01,gxefp1,gxm
      write(6,150) rank,gysf,gysf1,gysf2,gyefm1,gyef01,gyefp1,gym
      write(6,160) rank,gzsf,gzsf1,gzsf2,gzefm1,gzef01,gzefp1,gzm
 140  format('Fort ghosts: rank ',i4,
     & ': gxsf,gxsf1,gxsf2,gxefm1,gxef01,gxefp1,gxm:',9(i4))
 150  format('Fort ghosts: rank ',i4,
     & ': gysf,gysf1,gysf2,gyefm1,gyef01,gyefp1,gym:',9(i4))
 160  format('Fort ghosts: rank ',i4,
     & ': gzsf,gzsf1,gzsf2,gzefm1,gzef01,gzefp1,gzm:',9(i4))
      write(6,170) rank,gxsfw,gxsf1w,gxsf2w,gxefw
      write(6,180) rank,gysfw,gysf1w,gysf2w,gyefw
      write(6,190) rank,gzsfw,gzsf1w,gzsf2w,gzefw
 170  format('Fort ghosts (interior): rank ',
     & i4,': gxsfw,gxsf1w,gxsf2w,gxefw:',4(i4))
 180  format('Fort ghosts (interior): rank ',
     & i4,': gysfw,gysf1w,gysf2w,gyefw:',4(i4))
 190  format('Fort ghosts (interior): rank ',
     & i4,': gzsfw,gzsf1w,gzsf2w,gzefw:',4(i4))
#ifdef PARCH_rs6000
      call FLUSH_(6)
#else
      call flush(6)
#endif
      call PetscSequentialPhaseEnd(comm,1,ierr)
      endif

      return
      end

c -------------------------------------------------------------------

      integer function jmonitor(xl2,cfl_new,work,xx,p,dxx,
     &                          aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                          tcl,tcd,nsup)
c
c jmonitor - Writes miscellaneous output information.  Called by
c            the MonitorEuler().

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

      double precision  drecv(3),dsend(3),rtmax,rtrms,etmax,etrms
      double precision  dsup,con,rt,et,rtrmsr,etrmsr,tcl,tcd,areas,cn,cc
      double precision  area,pbody,cls,cds,xl2,work(*),cfl_new
      integer i,j,k,ierr,nsup

      ncyc     = ncyc + 1
      jmonitor = 0
      cfl      = cfl_new

c      IF (NCYC.NE.1.AND.MOD(NCYC,5).NE.0) RETURN
c Compute and tabulate norms and the number of supersonic points in flowfield
      RTMAX=zero
      RTRMS=zero
      ETMAX=zero
      ETRMS=zero
      NSUP=0
      dsup=zero
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
      TCL=zero
      TCD=zero
      AREAS=zero

      if (ysf1 .eq. 1) then
        DO 41 K=zsf2,KTIPP
c        DO 41 K=2,KTIP
          CN=zero
          CC=zero
          AREA=zero
          DO 42 I=ITLP1,ITUP
c          DO 42 I=ITL+1,ITU
            AREA=AREA+ABS(AJY(I,1,K))*p5
            PBODY=p5*(P(I,1,K)+P(I,2,K))
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
        dsend(1) = zero
        dsend(2) = zero
        dsend(3) = zero
      endif
      call MPI_Allreduce(dsend,drecv,3,MPI_DOUBLE_PRECISION,
     &                   MPI_SUM,comm,ierr)
      tcl   = drecv(1)
      tcd   = drecv(2)
      areas = drecv(3)
      TCL=TCL/AREAS*2./FSMACH**2
      TCD=TCD/AREAS*2./FSMACH**2

      if (no_output .ne. 0) return
      IF (NCYC.NE.1.AND.MOD(NCYC,5).NE.0) RETURN
      if (rank .eq. 0)
     &   WRITE (6,200) NCYC,RTMAX,RTRMS,ETRMS,XL2,TCL,TCD,NSUP

  200 FORMAT(I5,1P6E11.3,I5)
      RETURN
      END

c -------------------------------------------------------------------
c
      subroutine printjul(xx,p,base)

c printjul - Prints grid points (interior only)

#include "param.h"
#include "varbls.h"

      integer i,j,k
      integer unit, base

      if (base .lt. 10) base = 50
      unit = base + rank

      do 20 k=zsf2,zef01
      do 20 j=ysf2,yef01
      do 20 i=xsf2,xef01
            write(unit,100) k,j,i,R(I,J,K),RU(I,J,K),
     &            RV(i,j,k),RW(i,j,k),E(i,j,k),P(i,j,k)
   20    continue
  100 format(i4,i3,i3,6e14.6)
#ifdef PARCH_rs6000
      call FLUSH_(unit)
#else
      call flush(unit)
#endif

      return
      end

c -------------------------------------------------------------------
c
      subroutine printgjul(xx,p,base)

c printjul - Prints grid points (exterior/interior and ghost points)

#include "param.h"
#include "varbls.h"

      integer i,j,k
      integer unit, base

      if (base .lt. 10) base = 50
      unit = base + rank

      do 20 k=gzsf1,gzefp1
      do 20 j=gysf1,gyefp1
      do 20 i=gxsf1,gxefp1
            write(unit,100) k,j,i,R(i,j,k),
     &            RU(i,j,k),RV(i,j,k),
     &            RW(i,j,k),E(i,j,k),P(i,j,k)
   20    continue
  100 format(i4,i3,i3,6e14.6)
#ifdef PARCH_rs6000
      call FLUSH_(unit)
#else
      call flush(unit)
#endif

      return
      end

c -------------------------------------------------------------------
c
      subroutine printbjul(xx,p,base)

c printjul - Prints boundary grid points

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
     &          i.eq.ni1 .or. j.eq.nj1 .or. k.eq.nk1_boundary)
     &            write(unit,100) k,j,i,R(I,J,K),RU(I,J,K),
     &            rv(i,j,k),rw(i,j,k),e(i,j,k),p(i,j,k)
   20    continue
  100 format(i4,i3,i3,6e14.6)
#ifdef PARCH_rs6000
      call FLUSH_(unit)
#else
      call flush(unit)
#endif

      return
      end

c -------------------------------------------------------------------
c
c  readmesh - Reads mesh and wing parameters.
c
c  By default, we use these global arrays only temporarily for parallel
c  case.  Each processor reads all input data, and then later copies
c  only its local piece and frees the global mesh.  The input file for
c  a particular mesh size is opened in the routine parsetup().
c
      integer function readmesh(itlA,ituA,ileA,ktipA,x,y,z)

#include "param.h"
#include "wing.h"

      double precision  X(NI,NJ,NK),Y(NI,NJ,NK),Z(NI,NJ,NK)
      integer itlA, ituA, ileA, ktipA

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
