
      integer function PVAR(xx,p,
     &                      aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                      x,y,z,pprint,tcl,tcd,wcomm,wing)
c
c PVAR - post-processes the solution for physical quanities of interest
c This parallel version relies on the fact that currently each processor
c stores the entire mesh; then certain quantities (such as surface pressure)
c are directly computed and sent to all processors.   Only a single 
c processor prints the output.  Obviously, this procedure is suboptimal
c in terms of memory usage and communication.  A better variant may
c eventually replace this.  The uniprocessor analog is pvar_uni().
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
      Double  CP2(D_NI,D_NK),CP21(D_NI,D_NK)
      Double  CN1(D_NK),CC1(D_NK),CN(D_NK),CC(D_NK)
      Double  AREA(D_NK),AREA1(D_NK),ZLOC(D_NK)
      Double  S(D_NI),CL(D_NI),CD(D_NI)
      Double  chord,pbody,tcl,tcd,areas,cls,cds
      integer i,k,pprint,ierr,btot,wcomm,wing

      pvar = 0

      do 5 k=1,d_nk
      do 5 i=1,d_ni
 5         cp21(i,k) = 0.0

      if (ysf1 .eq. 1) then
         DO 10 K=zsf2,KTIPP
c         DO 10 K=2,KTIP
            CHORD   = X(ITU,1,K)-X(ILE,1,K)
            DO 10 I=ITLP,ITUP1
c            DO 10 I=ITL,ITU+1
               PBODY     = p5*(P(I,1,K)+P(I,2,K))
               CP21(I,K) = (PBODY-RGAM)/FSMACH**2*2.
               S(I)      = ((X(I,1,K)+X(I-1,1,K))
     &                     *p5-X(ILE,1,K))/CHORD
 10      CONTINUE
      endif

      btot = d_nk*d_ni
      call MPI_Allreduce(cp21,cp2,btot,MPI_DOUBLE_PRECISION,
     &                       MPI_SUM,comm,ierr)

      if (rank .eq. 0) then
         DO 15 K=2,KTIP
            ZLOC(K) = (Z(1,1,K)+Z(1,1,K-1))*p5
            CHORD   = X(ITU,1,K)-X(ILE,1,K)
            DO 15 I=ITL,ITU+1
               S(I)      = ((X(I,1,K)+X(I-1,1,K))
     &                     *p5-X(ILE,1,K))/CHORD
 15      CONTINUE

      IF (NI.EQ.49) THEN
         WRITE (6,20) NCYC
         WRITE (6,21) (ZLOC(K),K=2,KTIP)  
         WRITE (6,30) (S(I),(CP2(I,K),K=2,KTIP),I=ITL,ITU+1)
      ELSE IF (NI.EQ.97) THEN
         WRITE (6,90) NCYC
         WRITE (6,91) (ZLOC(K),K=2,KTIP)
         WRITE (6,93) (S(I),(CP2(I,K),K=2,KTIP),I=ITL,ITU+1)
      ELSE IF (NI.EQ.193) THEN
         WRITE (6,80) NCYC
         WRITE (6,81) (ZLOC(K),K=2,KTIP)
         WRITE (6,83) (S(I),(CP2(I,K),K=2,KTIP),I=ITL,ITU+1)
      ENDIF
      endif

   20 FORMAT (/,4X,'WING SURFACE PRESSURE COEFFICIENTS FOR',
     * ' NCYC =',I6,/,4X,'X/C',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP')
   21 FORMAT (4X,'Z = ',5F8.4,/)
   30 FORMAT(6F8.4)

   90 FORMAT (/,4X,'WING SURFACE PRESSURE COEFFICIENTS FOR',
     * ' NCYC =',I6,/,4X,'X/C',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP')
   91 FORMAT (4X,'Z = ',10F8.4,/)
   93 FORMAT(11F8.4)

   80 FORMAT (/,4X,'WING SURFACE PRESSURE COEFFICIENTS FOR',
     * ' NCYC =',I6,/,4X,'X/C',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP')
   81 FORMAT (4X,'Z = ',20F8.4,/)
   83 FORMAT(21F8.4)

      if (ysf1 .eq. 1) then
        DO 41 K=zsf2,KTIPP
c        DO 41 K=2,KTIP
          CN1(K)=zero
          CN(K)=zero
          CC1(K)=zero
          CC(K)=zero
          AREA1(K)=zero
          AREA(K)=zero
          DO 41 I=ITLP1,ITUP
c          DO 41 I=ITL+1,ITU
            AREA1(K)=AREA1(K)+ABS(AJY(I,1,K))*p5
            PBODY=p5*(P(I,1,K)+P(I,2,K))
            CN1(K)=CN1(K)-(PBODY-RGAM)*AJY(I,1,K)/FSMACH**2*2.
            CC1(K)=CC1(K)-(PBODY-RGAM)*AJX(I,1,K)/FSMACH**2*2.
c            CN=CN-(PBODY-RGAM)*AJY(I,1,K)
c            CC=CC-(PBODY-RGAM)*AJX(I,1,K)
 41     CONTINUE

c  Sum over each section of mesh, using the communicator wcomm
        if (wing .ne. 0) then
          call MPI_Allreduce(cn1,cn,d_nk,MPI_DOUBLE_PRECISION,
     &                       MPI_SUM,wcomm,ierr)
          call MPI_Allreduce(cc1,cc,d_nk,MPI_DOUBLE_PRECISION,
     &                       MPI_SUM,wcomm,ierr)
          call MPI_Allreduce(area1,area,d_nk,MPI_DOUBLE_PRECISION,
     &                       MPI_SUM,wcomm,ierr)
        endif

        if (wing .eq. 2) then
          TCL=zero
          TCD=zero
          AREAS=zero
          DO 42 K=2,KTIP
            CLS=CN(K)*COS(-BETA)-CC(K)*SIN(-BETA)
            CDS=CN(K)*SIN(-BETA)+CC(K)*COS(-BETA)
            CL(K) = CLS/AREA(K)
            CD(K) = CDS/AREA(K)
            AREAS = AREAS+AREA(K)
            TCL   = TCL+CLS
            TCD   = TCD+CDS
 42       CONTINUE
          TCL = TCL/AREAS
          TCD = TCD/AREAS
          WRITE (6,43)
          WRITE (6,44)
          WRITE (6,45) (ZLOC(K),CL(K),CD(K),K=2,KTIP)
          WRITE (6,46) TCL,TCD
        endif
      endif

   43 FORMAT (/ /,20X,'SPANWISE LIFT DISTRIBUTION',/ /)
   44 FORMAT (5X,'Z',12X,'CL',12X,'CD',/)
   45 FORMAT (2X,F6.3,4X,F10.4,4X,F10.4)
   46 FORMAT (/,2X,'WING LIFT COEFFICIENT =',F10.4,5X,
     1 'WING DRAG COEFFICIENT =',F10.4,/)

      RETURN
      END

c -------------------------------------------------------------------
c
c  wingsurface - Determines whether this processor's local mesh
c                includes part of the wing surface.  This routine
c                is called by GetWingCommunicator() in preparation
c                for generating a subsidiary communicator that
c                is used in pvar().
c
c  Output Parameter:
c    surface - 2 if this processor owns the point (itl1,2,2)
c              1 if this processor owns any other part of wing surface;
c              0 otherwise
c
      integer function wingsurface(surface)

#include "param.h"
#include "wing.h"

      integer surface,itl1

      wingsurface = 0

      if (zsf2 .gt. ktip .or. ysf2 .ne. 2) then
        surface = 0
      else
        itl1 = itl+1
        if (itl1 .le. xsf2) then
          if (itu .ge. xsf2) then
            surface = 1
          else
            surface = 0
          endif
        else 
          if (itl1 .le. xef01) then
            surface = 1
          else
            surface = 0
          endif
        endif
        if (zsf2 .eq. 2 .and. itl1 .ge. xsf2 .and.
     &      itl1 .le. xef01) surface = 2
      endif

      if (printg .eq. 1) then
        if (surface .ne. 0) then
          write(6,*) 'part of wing surface: rank',rank
          if (surface .eq. 2) 
     &       write(6,*) 'own point (itl1,2,2): rank',rank
        else 
          write(6,*) 'NOT part of wing surface: rank ',rank
        endif
#ifdef PARCH_rs6000
        call FLUSH_(6)
#else
        call flush(6)
#endif
      endif

      return
      end
