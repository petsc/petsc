C
C  $Id: ts.h,v 1.7 1997/11/14 14:59:45 bsmith Exp bsmith $;
C
C  Include file for Fortran use of the TS (timestepping) package in PETSc
C
#define TS            integer
#define TSProblemType integer 
#define TSPVodeType   integer

#define TS_EULER  'euler'
#define TS_BEULER 'beuler'
#define TS_PSEUDO 'pseudo'
#define TS_PVODE  'pvode'

      integer TS_LINEAR, TS_NONLINEAR
      parameter (TS_LINEAR = 0, TS_NONLINEAR = 1)

      integer PVODE_ADAMS, PVODE_BDF
      parameter (PVODE_ADAMS=0, PVODE_BDF=1)
C
C  End of Fortran include file for the TS package in PETSc

