/*
      SUBROUTINE HWSCRT (A,B,M,MBDCND,BDA,BDB,C,D,N,NBDCND,BDC,BDD,
     1                   ELMBDA,F,IDIMF,PERTRB,IERROR,W)
C
C
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C     *                                                               *
C     *                        F I S H P A K                          *
C     *                                                               *
C     *                                                               *
C     *     A PACKAGE OF FORTRAN SUBPROGRAMS FOR THE SOLUTION OF      *
C     *                                                               *
C     *      SEPARABLE ELLIPTIC PARTIAL DIFFERENTIAL EQUATIONS        *
C     *                                                               *
C     *                  (VERSION 3.1 , OCTOBER 1980)                  *
C     *                                                               *
C     *                             BY                                *
C     *                                                               *
C     *        JOHN ADAMS, PAUL SWARZTRAUBER AND ROLAND SWEET         *
C     *                                                               *
C     *                             OF                                *
C     *                                                               *
C     *         THE NATIONAL CENTER FOR ATMOSPHERIC RESEARCH          *
C     *                                                               *
C     *                BOULDER, COLORADO  (80307)  U.S.A.             *
C     *                                                               *
C     *                   WHICH IS SPONSORED BY                       *
C     *                                                               *
C     *              THE NATIONAL SCIENCE FOUNDATION                  *
C     *                                                               *
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C
C     * * * * * * * * *  PURPOSE    * * * * * * * * * * * * * * * * * *
C
C          SUBROUTINE HWSCRT SOLVES THE STANDARD FIVE-POINT FINITE
C     DIFFERENCE APPROXIMATION TO THE HELMHOLTZ EQUATION IN CARTESIAN
C     COORDINATES:
C
C          (D/DX)(DU/DX) + (D/DY)(DU/DY) + LAMBDA*U = F(X,Y).
C
C
C
C     * * * * * * * *    PARAMETER DESCRIPTION     * * * * * * * * * *
C
C             * * * * * *   ON INPUT    * * * * * *
C
C     A,B
C       THE RANGE OF X, I.E., A .LE. X .LE. B.  A MUST BE LESS THAN B.
C
C     M
C       THE NUMBER OF PANELS INTO WHICH THE INTERVAL (A,B) IS
C       SUBDIVIDED.  HENCE, THERE WILL BE M+1 GRID POINTS IN THE
C       X-DIRECTION GIVEN BY X(I) = A+(I-1)DX FOR I = 1,2,...,M+1,
C       WHERE DX = (B-A)/M IS THE PANEL WIDTH. M MUST BE GREATER THAN 3.
C
C     MBDCND
C       INDICATES THE TYPE OF BOUNDARY CONDITIONS AT X = A AND X = B.
C
C       = 0  IF THE SOLUTION IS PERIODIC IN X, I.E., U(I,J) = U(M+I,J).
C       = 1  IF THE SOLUTION IS SPECIFIED AT X = A AND X = B.
C       = 2  IF THE SOLUTION IS SPECIFIED AT X = A AND THE DERIVATIVE OF
C            THE SOLUTION WITH RESPECT TO X IS SPECIFIED AT X = B.
C       = 3  IF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO X IS
C            SPECIFIED AT X = A AND X = B.
C       = 4  IF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO X IS
C            SPECIFIED AT X = A AND THE SOLUTION IS SPECIFIED AT X = B.
C
C     BDA
C       A ONE-DIMENSIONAL ARRAY OF LENGTH N+1 THAT SPECIFIES THE VALUES
C       OF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO X AT X = A.
C       WHEN MBDCND = 3 OR 4,
C
C            BDA(J) = (D/DX)U(A,Y(J)), J = 1,2,...,N+1  .
C
C       WHEN MBDCND HAS ANY OTHER VALUE, BDA IS A DUMMY VARIABLE.
C
C     BDB
C       A ONE-DIMENSIONAL ARRAY OF LENGTH N+1 THAT SPECIFIES THE VALUES
C       OF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO X AT X = B.
C       WHEN MBDCND = 2 OR 3,
C
C            BDB(J) = (D/DX)U(B,Y(J)), J = 1,2,...,N+1  .
C
C       WHEN MBDCND HAS ANY OTHER VALUE BDB IS A DUMMY VARIABLE.
C
C     C,D
C       THE RANGE OF Y, I.E., C .LE. Y .LE. D.  C MUST BE LESS THAN D.
C
C     N
C       THE NUMBER OF PANELS INTO WHICH THE INTERVAL (C,D) IS
C       SUBDIVIDED.  HENCE, THERE WILL BE N+1 GRID POINTS IN THE
C       Y-DIRECTION GIVEN BY Y(J) = C+(J-1)DY FOR J = 1,2,...,N+1, WHERE
C       DY = (D-C)/N IS THE PANEL WIDTH.  N MUST BE GREATER THAN 3.
C
C     NBDCND
C       INDICATES THE TYPE OF BOUNDARY CONDITIONS AT Y = C AND Y = D.
C
C       = 0  IF THE SOLUTION IS PERIODIC IN Y, I.E., U(I,J) = U(I,N+J).
C       = 1  IF THE SOLUTION IS SPECIFIED AT Y = C AND Y = D.
C       = 2  IF THE SOLUTION IS SPECIFIED AT Y = C AND THE DERIVATIVE OF
C            THE SOLUTION WITH RESPECT TO Y IS SPECIFIED AT Y = D.
C       = 3  IF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO Y IS
C            SPECIFIED AT Y = C AND Y = D.
C       = 4  IF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO Y IS
C            SPECIFIED AT Y = C AND THE SOLUTION IS SPECIFIED AT Y = D.
C
C     BDC
C       A ONE-DIMENSIONAL ARRAY OF LENGTH M+1 THAT SPECIFIES THE VALUES
C       OF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO Y AT Y = C.
C       WHEN NBDCND = 3 OR 4,
C
C            BDC(I) = (D/DY)U(X(I),C), I = 1,2,...,M+1  .
C
C       WHEN NBDCND HAS ANY OTHER VALUE, BDC IS A DUMMY VARIABLE.
C
C     BDD
C       A ONE-DIMENSIONAL ARRAY OF LENGTH M+1 THAT SPECIFIES THE VALUES
C       OF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO Y AT Y = D.
C       WHEN NBDCND = 2 OR 3,
C
C            BDD(I) = (D/DY)U(X(I),D), I = 1,2,...,M+1  .
C
C       WHEN NBDCND HAS ANY OTHER VALUE, BDD IS A DUMMY VARIABLE.
C
C     ELMBDA
C       THE CONSTANT LAMBDA IN THE HELMHOLTZ EQUATION.  IF
C       LAMBDA .GT. 0, A SOLUTION MAY NOT EXIST.  HOWEVER, HWSCRT WILL
C       ATTEMPT TO FIND A SOLUTION.
C
C     F
C       A TWO-DIMENSIONAL ARRAY WHICH SPECIFIES THE VALUES OF THE RIGHT
C       SIDE OF THE HELMHOLTZ EQUATION AND BOUNDARY VALUES (IF ANY).
C       FOR I = 2,3,...,M AND J = 2,3,...,N
C
C            F(I,J) = F(X(I),Y(J)).
C
C       ON THE BOUNDARIES F IS DEFINED BY
C
C            MBDCND     F(1,J)        F(M+1,J)
C            ------     ---------     --------
C
C              0        F(A,Y(J))     F(A,Y(J))
C              1        U(A,Y(J))     U(B,Y(J))
C              2        U(A,Y(J))     F(B,Y(J))     J = 1,2,...,N+1
C              3        F(A,Y(J))     F(B,Y(J))
C              4        F(A,Y(J))     U(B,Y(J))
C
C
C            NBDCND     F(I,1)        F(I,N+1)
C            ------     ---------     --------
C
C              0        F(X(I),C)     F(X(I),C)
C              1        U(X(I),C)     U(X(I),D)
C              2        U(X(I),C)     F(X(I),D)     I = 1,2,...,M+1
C              3        F(X(I),C)     F(X(I),D)
C              4        F(X(I),C)     U(X(I),D)
C
C       F MUST BE DIMENSIONED AT LEAST (M+1)*(N+1).
C
C       NOTE
C
C       IF THE TABLE CALLS FOR BOTH THE SOLUTION U AND THE RIGHT SIDE F
C       AT  A CORNER THEN THE SOLUTION MUST BE SPECIFIED.
C
C     IDIMF
C       THE ROW (OR FIRST) DIMENSION OF THE ARRAY F AS IT APPEARS IN THE
C       PROGRAM CALLING HWSCRT.  THIS PARAMETER IS USED TO SPECIFY THE
C       VARIABLE DIMENSION OF F.  IDIMF MUST BE AT LEAST M+1  .
C
C     W
C       A ONE-DIMENSIONAL ARRAY THAT MUST BE PROVIDED BY THE USER FOR
C       WORK SPACE.  W MAY REQUIRE UP TO 4*(N+1) +
C       (13 + INT(LOG2(N+1)))*(M+1) LOCATIONS.  THE ACTUAL NUMBER OF
C       LOCATIONS USED IS COMPUTED BY HWSCRT AND IS RETURNED IN LOCATION
C       W(1).
C
C
C             * * * * * *   ON OUTPUT     * * * * * *
C
C     F
C       CONTAINS THE SOLUTION U(I,J) OF THE FINITE DIFFERENCE
C       APPROXIMATION FOR THE GRID POINT (X(I),Y(J)), I = 1,2,...,M+1,
C       J = 1,2,...,N+1  .
C
C     PERTRB
C       IF A COMBINATION OF PERIODIC OR DERIVATIVE BOUNDARY CONDITIONS
C       IS SPECIFIED FOR A POISSON EQUATION (LAMBDA = 0), A SOLUTION MAY
C       NOT EXIST.  PERTRB IS A CONSTANT, CALCULATED AND SUBTRACTED FROM
C       F, WHICH ENSURES THAT A SOLUTION EXISTS.  HWSCRT THEN COMPUTES
C       THIS SOLUTION, WHICH IS A LEAST SQUARES SOLUTION TO THE ORIGINAL
C       APPROXIMATION.  THIS SOLUTION PLUS ANY CONSTANT IS ALSO A
C       SOLUTION.  HENCE, THE SOLUTION IS NOT UNIQUE.  THE VALUE OF
C       PERTRB SHOULD BE SMALL COMPARED TO THE RIGHT SIDE F.  OTHERWISE,
C       A SOLUTION IS OBTAINED TO AN ESSENTIALLY DIFFERENT PROBLEM.
C       THIS COMPARISON SHOULD ALWAYS BE MADE TO INSURE THAT A
C       MEANINGFUL SOLUTION HAS BEEN OBTAINED.
C
C     IERROR
C       AN ERROR FLAG THAT INDICATES INVALID INPUT PARAMETERS.  EXCEPT
C       FOR NUMBERS 0 AND 6, A SOLUTION IS NOT ATTEMPTED.
C
C       = 0  NO ERROR.
C       = 1  A .GE. B.
C       = 2  MBDCND .LT. 0 OR MBDCND .GT. 4  .
C       = 3  C .GE. D.
C       = 4  N .LE. 3
C       = 5  NBDCND .LT. 0 OR NBDCND .GT. 4  .
C       = 6  LAMBDA .GT. 0  .
C       = 7  IDIMF .LT. M+1  .
C       = 8  M .LE. 3
C
C       SINCE THIS IS THE ONLY MEANS OF INDICATING A POSSIBLY INCORRECT
C       CALL TO HWSCRT, THE USER SHOULD TEST IERROR AFTER THE CALL.
C
C     W
C       W(1) CONTAINS THE REQUIRED LENGTH OF W.
C
C
C     * * * * * * *   PROGRAM SPECIFICATIONS    * * * * * * * * * * * *
C
C
C     DIMENSION OF   BDA(N+1),BDB(N+1),BDC(M+1),BDD(M+1),F(IDIMF,N+1),
C     ARGUMENTS      W(SEE ARGUMENT LIST)
C
C     LATEST         JUNE 1, 1976
C     REVISION
C
C     SUBPROGRAMS    HWSCRT,GENBUN,POISD2,POISN2,POISP2,COSGEN,MERGE,
C     REQUIRED       TRIX,TRI3,PIMACH
C
C     SPECIAL        NONE
C     CONDITIONS
C
C     COMMON         NONE
C     BLOCKS
C
C     I/O            NONE
C
C     PRECISION      SINGLE
C
C     SPECIALIST     ROLAND SWEET
C
C     LANGUAGE       FORTRAN
C
C     HISTORY        STANDARDIZED SEPTEMBER 1, 1973
C                    REVISED APRIL 1, 1976
C
C     ALGORITHM      THE ROUTINE DEFINES THE FINITE DIFFERENCE
C                    EQUATIONS, INCORPORATES BOUNDARY DATA, AND ADJUSTS
C                    THE RIGHT SIDE OF SINGULAR SYSTEMS AND THEN CALLS
C                    GENBUN TO SOLVE THE SYSTEM.
C
C     SPACE          13110(OCTAL) = 5704(DECIMAL) LOCATIONS ON THE NCAR
C     REQUIRED       CONTROL DATA 7600
C
C     TIMING AND        THE EXECUTION TIME T ON THE NCAR CONTROL DATA
C     ACCURACY       7600 FOR SUBROUTINE HWSCRT IS ROUGHLY PROPORTIONAL
C                    TO M*N*LOG2(N), BUT ALSO DEPENDS ON THE INPUT
C                    PARAMETERS NBDCND AND MBDCND.  SOME TYPICAL VALUES
C                    ARE LISTED IN THE TABLE BELOW.
C                       THE SOLUTION PROCESS EMPLOYED RESULTS IN A LOSS
C                    OF NO MORE THAN THREE SIGNIFICANT DIGITS FOR N AND
C                    M AS LARGE AS 64.  MORE DETAILED INFORMATION ABOUT
C                    ACCURACY CAN BE FOUND IN THE DOCUMENTATION FOR
C                    SUBROUTINE GENBUN WHICH IS THE ROUTINE THAT
C                    SOLVES THE FINITE DIFFERENCE EQUATIONS.
C
C
C                       M(=N)    MBDCND    NBDCND    T(MSECS)
C                       -----    ------    ------    --------
C
C                        32        0         0          31
C                        32        1         1          23
C                        32        3         3          36
C                        64        0         0         128
C                        64        1         1          96
C                        64        3         3         142
C
C     PORTABILITY    AMERICAN NATIONAL STANDARDS INSTITUTE FORTRAN.
C                    ALL MACHINE DEPENDENT CONSTANTS ARE LOCATED IN THE
C                    FUNCTION PIMACH.
C
C     REFERENCE      SWARZTRAUBER,P. AND R. SWEET, 'EFFICIENT FORTRAN
C                    SUBPROGRAMS FOR THE SOLUTION OF ELLIPTIC EQUATIONS'
C                    NCAR TN/IA-109, JULY, 1975, 138 PP.
C
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C
*/

  void hwscrt_(double *a, double *b, int *m, int *mbdcnd, 
	       double *bda, double *bdb,
               double *c, double *d, int *n, int *nbdcnd, 
	       double *bdc, double *bdd,
               double *elmbda, double *f, int *idimf,
               double *pertrb, int *ierror, double *w);

/*
C
C
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C     *                                                               *
C     *                        F I S H P A K                          *
C     *                                                               *
C     *                                                               *
C     *     A PACKAGE OF FORTRAN SUBPROGRAMS FOR THE SOLUTION OF      *
C     *                                                               *
C     *      SEPARABLE ELLIPTIC PARTIAL DIFFERENTIAL EQUATIONS        *
C     *                                                               *
C     *                  (VERSION 3.1 , OCTOBER 1980)                  *
C     *                                                               *
C     *                             BY                                *
C     *                                                               *
C     *        JOHN ADAMS, PAUL SWARZTRAUBER AND ROLAND SWEET         *
C     *                                                               *
C     *                             OF                                *
C     *                                                               *
C     *         THE NATIONAL CENTER FOR ATMOSPHERIC RESEARCH          *
C     *                                                               *
C     *                BOULDER, COLORADO  (80307)  U.S.A.             *
C     *                                                               *
C     *                   WHICH IS SPONSORED BY                       *
C     *                                                               *
C     *              THE NATIONAL SCIENCE FOUNDATION                  *
C     *                                                               *
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C
C    * * * * * * * * *  PURPOSE    * * * * * * * * * * * * * * * * * *
C
C          SUBROUTINE HW3CRT SOLVES THE STANDARD SEVEN-POINT FINITE
C     DIFFERENCE APPROXIMATION TO THE HELMHOLTZ EQUATION IN CARTESIAN
C     COORDINATES:
C
C         (D/DX)(DU/DX) + (D/DY)(DU/DY) + (D/DZ)(DU/DZ)
C
C                    + LAMBDA*U = F(X,Y,Z) .
C
C    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C
C    * * * * * * * *    PARAMETER DESCRIPTION     * * * * * * * * * *
C
C
C            * * * * * *   ON INPUT    * * * * * *
C
C     XS,XF
C        THE RANGE OF X, I.E. XS .LE. X .LE. XF .
C        XS MUST BE LESS THAN XF.
C
C     L
C        THE NUMBER OF PANELS INTO WHICH THE INTERVAL (XS,XF) IS
C        SUBDIVIDED.  HENCE, THERE WILL BE L+1 GRID POINTS IN THE
C        X-DIRECTION GIVEN BY X(I) = XS+(I-1)DX FOR I=1,2,...,L+1,
C        WHERE DX = (XF-XS)/L IS THE PANEL WIDTH.  L MUST BE AT
C        LEAST 5 .
C
C     LBDCND
C        INDICATES THE TYPE OF BOUNDARY CONDITIONS AT X = XS AND X = XF.
C
C        = 0  IF THE SOLUTION IS PERIODIC IN X, I.E.
C             U(L+I,J,K) = U(I,J,K).
C        = 1  IF THE SOLUTION IS SPECIFIED AT X = XS AND X = XF.
C        = 2  IF THE SOLUTION IS SPECIFIED AT X = XS AND THE DERIVATIVE
C             OF THE SOLUTION WITH RESPECT TO X IS SPECIFIED AT X = XF.
C        = 3  IF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO X IS
C             SPECIFIED AT X = XS AND X = XF.
C        = 4  IF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO X IS
C             SPECIFIED AT X = XS AND THE SOLUTION IS SPECIFIED AT X=XF.
C
C     BDXS
C        A TWO-DIMENSIONAL ARRAY THAT SPECIFIES THE VALUES OF THE
C        DERIVATIVE OF THE SOLUTION WITH RESPECT TO X AT X = XS.
C        WHEN LBDCND = 3 OR 4,
C
C             BDXS(J,K) = (D/DX)U(XS,Y(J),Z(K)), J=1,2,...,M+1,
C                                                K=1,2,...,N+1.
C
C        WHEN LBDCND HAS ANY OTHER VALUE, BDXS IS A DUMMY VARIABLE.
C        BDXS MUST BE DIMENSIONED AT LEAST (M+1)*(N+1).
C
C     BDXF
C        A TWO-DIMENSIONAL ARRAY THAT SPECIFIES THE VALUES OF THE
C        DERIVATIVE OF THE SOLUTION WITH RESPECT TO X AT X = XF.
C        WHEN LBDCND = 2 OR 3,
C
C             BDXF(J,K) = (D/DX)U(XF,Y(J),Z(K)), J=1,2,...,M+1,
C                                                K=1,2,...,N+1.
C
C        WHEN LBDCND HAS ANY OTHER VALUE, BDXF IS A DUMMY VARIABLE.
C        BDXF MUST BE DIMENSIONED AT LEAST (M+1)*(N+1).
C
C     YS,YF
C        THE RANGE OF Y, I.E. YS .LE. Y .LE. YF.
C        YS MUST BE LESS THAN YF.
C
C     M
C        THE NUMBER OF PANELS INTO WHICH THE INTERVAL (YS,YF) IS
C        SUBDIVIDED.  HENCE, THERE WILL BE M+1 GRID POINTS IN THE
C        Y-DIRECTION GIVEN BY Y(J) = YS+(J-1)DY FOR J=1,2,...,M+1,
C        WHERE DY = (YF-YS)/M IS THE PANEL WIDTH.  M MUST BE AT
C        LEAST 5 .
C
C     MBDCND
C        INDICATES THE TYPE OF BOUNDARY CONDITIONS AT Y = YS AND Y = YF.
C
C        = 0  IF THE SOLUTION IS PERIODIC IN Y, I.E.
C             U(I,M+J,K) = U(I,J,K).
C        = 1  IF THE SOLUTION IS SPECIFIED AT Y = YS AND Y = YF.
C        = 2  IF THE SOLUTION IS SPECIFIED AT Y = YS AND THE DERIVATIVE
C             OF THE SOLUTION WITH RESPECT TO Y IS SPECIFIED AT Y = YF.
C        = 3  IF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO Y IS
C             SPECIFIED AT Y = YS AND Y = YF.
C        = 4  IF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO Y IS
C             SPECIFIED AT Y = YS AND THE SOLUTION IS SPECIFIED AT Y=YF.
C
C     BDYS
C        A TWO-DIMENSIONAL ARRAY THAT SPECIFIES THE VALUES OF THE
C        DERIVATIVE OF THE SOLUTION WITH RESPECT TO Y AT Y = YS.
C        WHEN MBDCND = 3 OR 4,
C
C             BDYS(I,K) = (D/DY)U(X(I),YS,Z(K)), I=1,2,...,L+1,
C                                                K=1,2,...,N+1.
C
C        WHEN MBDCND HAS ANY OTHER VALUE, BDYS IS A DUMMY VARIABLE.
C        BDYS MUST BE DIMENSIONED AT LEAST (L+1)*(N+1).
C
C     BDYF
C        A TWO-DIMENSIONAL ARRAY THAT SPECIFIES THE VALUES OF THE
C        DERIVATIVE OF THE SOLUTION WITH RESPECT TO Y AT Y = YF.
C        WHEN MBDCND = 2 OR 3,
C
C             BDYF(I,K) = (D/DY)U(X(I),YF,Z(K)), I=1,2,...,L+1,
C                                                K=1,2,...,N+1.
C
C        WHEN MBDCND HAS ANY OTHER VALUE, BDYF IS A DUMMY VARIABLE.
C        BDYF MUST BE DIMENSIONED AT LEAST (L+1)*(N+1).
C
C     ZS,ZF
C        THE RANGE OF Z, I.E. ZS .LE. Z .LE. ZF.
C        ZS MUST BE LESS THAN ZF.
C
C     N
C        THE NUMBER OF PANELS INTO WHICH THE INTERVAL (ZS,ZF) IS
C        SUBDIVIDED.  HENCE, THERE WILL BE N+1 GRID POINTS IN THE
C        Z-DIRECTION GIVEN BY Z(K) = ZS+(K-1)DZ FOR K=1,2,...,N+1,
C        WHERE DZ = (ZF-ZS)/N IS THE PANEL WIDTH.  N MUST BE AT LEAST 5.
C
C     NBDCND
C        INDICATES THE TYPE OF BOUNDARY CONDITIONS AT Z = ZS AND Z = ZF.
C
C        = 0  IF THE SOLUTION IS PERIODIC IN Z, I.E.
C             U(I,J,N+K) = U(I,J,K).
C        = 1  IF THE SOLUTION IS SPECIFIED AT Z = ZS AND Z = ZF.
C        = 2  IF THE SOLUTION IS SPECIFIED AT Z = ZS AND THE DERIVATIVE
C             OF THE SOLUTION WITH RESPECT TO Z IS SPECIFIED AT Z = ZF.
C        = 3  IF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO Z IS
C             SPECIFIED AT Z = ZS AND Z = ZF.
C        = 4  IF THE DERIVATIVE OF THE SOLUTION WITH RESPECT TO Z IS
C             SPECIFIED AT Z = ZS AND THE SOLUTION IS SPECIFIED AT Z=ZF.
C
C     BDZS
C        A TWO-DIMENSIONAL ARRAY THAT SPECIFIES THE VALUES OF THE
C        DERIVATIVE OF THE SOLUTION WITH RESPECT TO Z AT Z = ZS.
C        WHEN NBDCND = 3 OR 4,
C
C             BDZS(I,J) = (D/DZ)U(X(I),Y(J),ZS), I=1,2,...,L+1,
C                                                J=1,2,...,M+1.
C
C        WHEN NBDCND HAS ANY OTHER VALUE, BDZS IS A DUMMY VARIABLE.
C        BDZS MUST BE DIMENSIONED AT LEAST (L+1)*(M+1).
C
C     BDZF
C        A TWO-DIMENSIONAL ARRAY THAT SPECIFIES THE VALUES OF THE
C        DERIVATIVE OF THE SOLUTION WITH RESPECT TO Z AT Z = ZF.
C        WHEN NBDCND = 2 OR 3,
C
C             BDZF(I,J) = (D/DZ)U(X(I),Y(J),ZF), I=1,2,...,L+1,
C                                                J=1,2,...,M+1.
C
C        WHEN NBDCND HAS ANY OTHER VALUE, BDZF IS A DUMMY VARIABLE.
C        BDZF MUST BE DIMENSIONED AT LEAST (L+1)*(M+1).
C
C     ELMBDA
C        THE CONSTANT LAMBDA IN THE HELMHOLTZ EQUATION. IF
C        LAMBDA .GT. 0, A SOLUTION MAY NOT EXIST.  HOWEVER, HW3CRT WILL
C        ATTEMPT TO FIND A SOLUTION.
C
C     F
C        A THREE-DIMENSIONAL ARRAY THAT SPECIFIES THE VALUES OF THE
C        RIGHT SIDE OF THE HELMHOLTZ EQUATION AND BOUNDARY VALUES (IF
C        ANY).  FOR I=2,3,...,L, J=2,3,...,M, AND K=2,3,...,N
C
C                   F(I,J,K) = F(X(I),Y(J),Z(K)).
C
C        ON THE BOUNDARIES F IS DEFINED BY
C
C        LBDCND      F(1,J,K)         F(L+1,J,K)
C        ------   ---------------   ---------------
C
C          0      F(XS,Y(J),Z(K))   F(XS,Y(J),Z(K))
C          1      U(XS,Y(J),Z(K))   U(XF,Y(J),Z(K))
C          2      U(XS,Y(J),Z(K))   F(XF,Y(J),Z(K))   J=1,2,...,M+1
C          3      F(XS,Y(J),Z(K))   F(XF,Y(J),Z(K))   K=1,2,...,N+1
C          4      F(XS,Y(J),Z(K))   U(XF,Y(J),Z(K))
C
C        MBDCND      F(I,1,K)         F(I,M+1,K)
C        ------   ---------------   ---------------
C
C          0      F(X(I),YS,Z(K))   F(X(I),YS,Z(K))
C          1      U(X(I),YS,Z(K))   U(X(I),YF,Z(K))
C          2      U(X(I),YS,Z(K))   F(X(I),YF,Z(K))   I=1,2,...,L+1
C          3      F(X(I),YS,Z(K))   F(X(I),YF,Z(K))   K=1,2,...,N+1
C          4      F(X(I),YS,Z(K))   U(X(I),YF,Z(K))
C
C        NBDCND      F(I,J,1)         F(I,J,N+1)
C        ------   ---------------   ---------------
C
C          0      F(X(I),Y(J),ZS)   F(X(I),Y(J),ZS)
C          1      U(X(I),Y(J),ZS)   U(X(I),Y(J),ZF)
C          2      U(X(I),Y(J),ZS)   F(X(I),Y(J),ZF)   I=1,2,...,L+1
C          3      F(X(I),Y(J),ZS)   F(X(I),Y(J),ZF)   J=1,2,...,M+1
C          4      F(X(I),Y(J),ZS)   U(X(I),Y(J),ZF)
C
C        F MUST BE DIMENSIONED AT LEAST (L+1)*(M+1)*(N+1).
C
C        NOTE:
C
C        IF THE TABLE CALLS FOR BOTH THE SOLUTION U AND THE RIGHT SIDE F
C        ON A BOUNDARY, THEN THE SOLUTION MUST BE SPECIFIED.
C
C     LDIMF
C        THE ROW (OR FIRST) DIMENSION OF THE ARRAYS F,BDYS,BDYF,BDZS,
C        AND BDZF AS IT APPEARS IN THE PROGRAM CALLING HW3CRT. THIS
C        PARAMETER IS USED TO SPECIFY THE VARIABLE DIMENSION OF THESE
C        ARRAYS.  LDIMF MUST BE AT LEAST L+1.
C
C     MDIMF
C        THE COLUMN (OR SECOND) DIMENSION OF THE ARRAY F AND THE ROW (OR
C        FIRST) DIMENSION OF THE ARRAYS BDXS AND BDXF AS IT APPEARS IN
C        THE PROGRAM CALLING HW3CRT.  THIS PARAMETER IS USED TO SPECIFY
C        THE VARIABLE DIMENSION OF THESE ARRAYS.
C        MDIMF MUST BE AT LEAST M+1.
C
C     W
C        A ONE-DIMENSIONAL ARRAY THAT MUST BE PROVIDED BY THE USER FOR
C        WORK SPACE.  THE LENGTH OF W MUST BE AT LEAST 30 + L + M + 5*N
C        + MAX(L,M,N) + 7*(INT((L+1)/2) + INT((M+1)/2))
C
C
C            * * * * * *   ON OUTPUT   * * * * * *
C
C     F
C        CONTAINS THE SOLUTION U(I,J,K) OF THE FINITE DIFFERENCE
C        APPROXIMATION FOR THE GRID POINT (X(I),Y(J),Z(K)) FOR
C        I=1,2,...,L+1, J=1,2,...,M+1, AND K=1,2,...,N+1.
C
C     PERTRB
C        IF A COMBINATION OF PERIODIC OR DERIVATIVE BOUNDARY CONDITIONS
C        IS SPECIFIED FOR A POISSON EQUATION (LAMBDA = 0), A SOLUTION
C        MAY NOT EXIST.  PERTRB IS A CONSTANT, CALCULATED AND SUBTRACTED
C        FROM F, WHICH ENSURES THAT A SOLUTION EXISTS.  PWSCRT THEN
C        COMPUTES THIS SOLUTION, WHICH IS A LEAST SQUARES SOLUTION TO
C        THE ORIGINAL APPROXIMATION.  THIS SOLUTION IS NOT UNIQUE AND IS
C        UNNORMALIZED.  THE VALUE OF PERTRB SHOULD BE SMALL COMPARED TO
C        THE RIGHT SIDE F.  OTHERWISE, A SOLUTION IS OBTAINED TO AN
C        ESSENTIALLY DIFFERENT PROBLEM.  THIS COMPARISON SHOULD ALWAYS
C        BE MADE TO INSURE THAT A MEANINGFUL SOLUTION HAS BEEN OBTAINED.
C
C     IERROR
C        AN ERROR FLAG THAT INDICATES INVALID INPUT PARAMETERS.  EXCEPT
C        FOR NUMBERS 0 AND 12, A SOLUTION IS NOT ATTEMPTED.
C
C        =  0  NO ERROR
C        =  1  XS .GE. XF
C        =  2  L .LT. 5
C        =  3  LBDCND .LT. 0 .OR. LBDCND .GT. 4
C        =  4  YS .GE. YF
C        =  5  M .LT. 5
C        =  6  MBDCND .LT. 0 .OR. MBDCND .GT. 4
C        =  7  ZS .GE. ZF
C        =  8  N .LT. 5
C        =  9  NBDCND .LT. 0 .OR. NBDCND .GT. 4
C        = 10  LDIMF .LT. L+1
C        = 11  MDIMF .LT. M+1
C        = 12  LAMBDA .GT. 0
C
C        SINCE THIS IS THE ONLY MEANS OF INDICATING A POSSIBLY INCORRECT
C        CALL TO HW3CRT, THE USER SHOULD TEST IERROR AFTER THE CALL.
C
C
C    * * * * * * *   PROGRAM SPECIFICATIONS    * * * * * * * * * * * *
C
C     DIMENSION OF   BDXS(MDIMF,N+1),BDXF(MDIMF,N+1),BDYS(LDIMF,N+1),
C     ARGUMENTS      BDYF(LDIMF,N+1),BDZS(LDIMF,M+1),BDZF(LDIMF,M+1),
C                    F(LDIMF,MDIMF,N+1),W(SEE ARGUMENT LIST)
C
C     LATEST         DECEMBER 1, 1978
C     REVISION
C
C     SUBPROGRAMS    HW3CRT,POIS3D,POS3D1,TRID,RFFTI,RFFTF,RFFTF1,
C     REQUIRED       RFFTB,RFFTB1,COSTI,COST,SINTI,SINT,COSQI,COSQF,
C                    COSQF1,COSQB,COSQB1,SINQI,SINQF,SINQB,CFFTI,
C                    CFFTI1,CFFTB,CFFTB1,PASSB2,PASSB3,PASSB4,PASSB,
C                    CFFTF,CFFTF1,PASSF1,PASSF2,PASSF3,PASSF4,PASSF,
C                    PIMACH
C
C     SPECIAL        NONE
C     CONDITIONS
C
C     COMMON         VALUE
C     BLOCKS
C
C     I/O            NONE
C
C     PRECISION      SINGLE
C
C     SPECIALIST     ROLAND SWEET
C
C     LANGUAGE       FORTRAN
C
C     HISTORY        WRITTEN BY ROLAND SWEET AT NCAR IN JULY,1977
C
C     ALGORITHM      THIS SUBROUTINE DEFINES THE FINITE DIFFERENCE
C                    EQUATIONS, INCORPORATES BOUNDARY DATA, AND
C                    ADJUSTS THE RIGHT SIDE OF SINGULAR SYSTEMS AND
C                    THEN CALLS POIS3D TO SOLVE THE SYSTEM.
C
C     SPACE          7862(DECIMAL) = 17300(OCTAL) LOCATIONS ON THE
C     REQUIRED       NCAR CONTROL DATA 7600
C
C     TIMING AND        THE EXECUTION TIME T ON THE NCAR CONTROL DATA
C     ACCURACY       7600 FOR SUBROUTINE HW3CRT IS ROUGHLY PROPORTIONAL
C                    TO L*M*N*(LOG2(L)+LOG2(M)+5), BUT ALSO DEPENDS ON
C                    INPUT PARAMETERS LBDCND AND MBDCND.  SOME TYPICAL
C                    VALUES ARE LISTED IN THE TABLE BELOW.
C                       THE SOLUTION PROCESS EMPLOYED RESULTS IN A LOSS
C                    OF NO MORE THAN THREE SIGNIFICANT DIGITS FOR L,M AN
C                    N AS LARGE AS 32.  MORE DETAILED INFORMATION ABOUT
C                    ACCURACY CAN BE FOUND IN THE DOCUMENTATION FOR
C                    SUBROUTINE POIS3D WHICH IS THE ROUTINE THAT ACTUALL
C                    SOLVES THE FINITE DIFFERENCE EQUATIONS.
C
C
C                       L(=M=N)     LBDCND(=MBDCND=NBDCND)      T(MSECS)
C                       -------     ----------------------      --------
C
C                         16                  0                    300
C                         16                  1                    302
C                         16                  3                    348
C                         32                  0                   1925
C                         32                  1                   1929
C                         32                  3                   2109
C
C     PORTABILITY    AMERICAN NATIONAL STANDARDS INSTITUTE FORTRAN.
C                    THE MACHINE DEPENDENT CONSTANT PI IS DEFINED IN
C                    FUNCTION PIMACH.
C
C     REQUIRED       COS,SIN,ATAN
C     RESIDENT
C     ROUTINES
C
C     REFERENCE      NONE
C
C     REQUIRED         COS,SIN,ATAN
C     RESIDENT
C     ROUTINES
C
C     REFERENCE        NONE
C
C    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
*/

  void hw3crt_(double *xs, double *xf, int *l, int *lbdcnd, double *bdxs, double *bdxf,
	       double *ys, double *yf, int *m, int *mbdcnd, double *bdys, double *bdyf,
	       double *zs, double *zf, int *n, int *nbdcnd, double *bdzs, double *bdzf,
	       double *elmbda, int *ldimf, int *mdimf, double *f, 
	       double *pertrb, int *ierror, double *w);
