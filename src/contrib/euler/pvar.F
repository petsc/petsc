
      integer function PVAR(xx,p,
     &                      aix,ajx,akx,aiy,ajy,aky,aiz,ajz,akz,
     &                      x,y,z,pprint,tcl,tcd)
c
c PVAR - post-processes the solution for physical quanities of interest
c This parallel version relies on the fact that currently each processor
c stores the entire mesh; then certain quantities (such as surface pressure)
c are directly computed and sent to all processors.   Only a single 
c processor prints the output.  Obviously, this procedure is far from
c optimal in terms of memory usage and communication.  A better variant
c may eventually relace this.  The uniprocessor analog is pvar_uni().
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
      Double  dsend(3),drecv(3),S(D_NI),CL(D_NI),CD(D_NI)
      Double  chord,pbody,tcl,tcd,areas,cls,cds
      integer i,k,pprint,ierr,btot,xcomm

      pvar = 0

      if (pprint .eq. 1) then
      do 5 k=1,d_nk
      do 5 i=1,d_ni
 5         cp21(i,k) = 0.0

      if (ysf1 .eq. 1) then
         DO 10 K=zsf2,KTIPP
c         DO 10 K=2,KTIP
            ZLOC(K) = (Z(1,1,K)+Z(1,1,K-1))*p5
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
      call MPI_Allreduce(cp21,btot,cp2,btot,MPI_DOUBLE_PRECISION,
     &                       MPI_SUM,comm,ierr)

      if (rank .eq. 0) then
      IF (NI.EQ.49) WRITE (6,20) NCYC
   20 FORMAT (/,4X,'WING SURFACE PRESSURE COEFFICIENTS FOR',
     * ' NCYC =',I6,/,4X,'X/C',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP')
      IF (NI.EQ.49) WRITE (6,21) (ZLOC(K),K=2,KTIP)
   21 FORMAT (4X,'Z = ',5F8.4,/)
      IF (NI.EQ.49)
     * WRITE (6,30) (S(I),(CP2(I,K),K=2,KTIP),I=ITL,ITU+1)
   30 FORMAT(6F8.4)
      IF (NI.EQ.97) WRITE (6,90) NCYC
   90 FORMAT (/,4X,'WING SURFACE PRESSURE COEFFICIENTS FOR',
     * ' NCYC =',I6,/,4X,'X/C',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP',6X,
     * 'CP',6X,'CP',6X,'CP',6X,'CP',6X,'CP')
      IF (NI.EQ.97) WRITE (6,91) (ZLOC(K),K=2,KTIP)
   91 FORMAT (4X,'Z = ',10F8.4,/)
      IF (NI.EQ.97)
     * WRITE (6,93) (S(I),(CP2(I,K),K=2,KTIP),I=ITL,ITU+1)
   93 FORMAT(11F8.4)
      endif
      endif

      return 

c
c  We MUST generate xcomm!
c
      TCL=zero
      TCD=zero
      AREAS=zero
      if (ysf1 .eq. 1) then
        DO 41 K=zsf2,KTIPP
c        DO 41 K=2,KTIP
          CN1(K)=zero
          CC1(K)=zero
          AREA1(K)=zero
          DO 42 I=ITLP1,ITUP
c          DO 42 I=ITL+1,ITU
            AREA1(K)=AREA1(K)+ABS(AJY(I,1,K))*p5
            PBODY=p5*(P(I,1,K)+P(I,2,K))
            CN1(K)=CN1(K)-(PBODY-RGAM)*AJY(I,1,K)/FSMACH**2*2.
            CC1(K)=CC1(K)-(PBODY-RGAM)*AJX(I,1,K)/FSMACH**2*2.
c            CN=CN-(PBODY-RGAM)*AJY(I,1,K)
c            CC=CC-(PBODY-RGAM)*AJX(I,1,K)
 42       CONTINUE
          call MPI_Allreduce(cn1,cn,d_nk,MPI_DOUBLE_PRECISION,
     &                       MPI_SUM,xcomm,ierr)
          call MPI_Allreduce(cc1,cc,d_nk,MPI_DOUBLE_PRECISION,
     &                       MPI_SUM,xcomm,ierr)
          call MPI_Allreduce(area1,area,d_nk,MPI_DOUBLE_PRECISION,
     &                       MPI_SUM,xcomm,ierr)
          CLS=CN(K)*COS(-BETA)-CC(K)*SIN(-BETA)
          CDS=CN(K)*SIN(-BETA)+CC(K)*COS(-BETA)
          if (size .eq. 1) then
            CL(K) = CLS/AREA(K)
            CD(K) = CDS/AREA(K)
          endif
          AREAS=AREAS+AREA(K)
          TCL=TCL+CLS
          TCD=TCD+CDS
 41     CONTINUE
        if (zsf .eq. 1) then
           dsend(1) = tcl
           dsend(2) = tcd
           dsend(3) = areas
        else
           dsend(1) = zero
           dsend(2) = zero
           dsend(3) = zero
        endif
      else
        dsend(1) = zero
        dsend(2) = zero
        dsend(3) = zero
      endif
        write(6,*) 'rank, tcl, tcd, areas = ',rank,tcl,tcd,areas

      call MPI_Allreduce(dsend,drecv,3,MPI_DOUBLE_PRECISION,
     &                   MPI_SUM,comm,ierr)
      tcl   = drecv(1)
      tcd   = drecv(2)
      areas = drecv(3)
      if (rank .eq. 0)
     & write(6,*) 'TOTAL: tcl, tcd, areas = ',tcl,tcd,areas

      TCL       = TCL/AREAS
      TCD       = TCD/AREAS

      if (rank .eq. 0)
     & write(6,*) 'TOTAL B: tcl, tcd, areas = ',tcl,tcd,areas

      if (pprint .eq. 1 .and. rank .eq. 0) then
      WRITE (6,43)
   43 FORMAT (/ /,20X,'SPANWISE LIFT DISTRIBUTION',/ /)
      WRITE (6,44)
   44 FORMAT (5X,'Z',12X,'CL',12X,'CD',/)
      WRITE (6,45) (ZLOC(K),CL(K),CD(K),K=2,KTIP)
   45 FORMAT (2X,F6.3,4X,F10.4,4X,F10.4)
      WRITE (6,46) TCL,TCD
   46 FORMAT (/,2X,'WING LIFT COEFFICIENT =',F10.4,5X,
     1 'WING DRAG COEFFICIENT =',F10.4,/)
      endif

      RETURN
      END
