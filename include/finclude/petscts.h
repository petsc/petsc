C
C  $Id: ts.h,v 1.5 1997/08/07 14:43:52 bsmith Exp balay $;
C
C  Include file for Fortran use of the TS (timestepping) package in PETSc
C
#define TS            integer
#define TSType        integer
#define TSProblemType integer 
#define TSPVodeType   integer

      integer TS_EULER, TS_BEULER, TS_PSEUDO, TS_PVODE, TS_NEW
      parameter (TS_EULER = 0, TS_BEULER = 1,TS_PSEUDO = 2,TS_PVODE = 3
     *           TS_NEW = 4 )

      integer TS_LINEAR, TS_NONLINEAR
      parameter (TS_LINEAR = 0, TS_NONLINEAR = 1)

      integer PVODE_ADAMS, PVODE_BDF
      parameter (PVODE_ADAMS=0, PVODE_BDF=1)
C
C  End of Fortran include file for the TS package in PETSc

