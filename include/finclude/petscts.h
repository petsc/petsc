C
C  $Id: ts.h,v 1.3 1996/09/30 20:19:11 curfman Exp curfman $;
C
C  Include file for Fortran use of the TS (timestepping) package in PETSc
C
#define TS            integer
#define TSType        integer
#define TSProblemType integer 

      integer TS_EULER, TS_BEULER, TS_PSEUDO

      parameter (TS_EULER = 0, TS_BEULER = 1,TS_PSEUDO = 2)

      integer TS_LINEAR, TS_NONLINEAR

      parameter (TS_LINEAR = 0, TS_NONLINEAR = 1)
C
C  End of Fortran include file for the TS package in PETSc

