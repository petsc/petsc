!
!  $Id: ts.h,v 1.8 1998/03/06 00:21:29 bsmith Exp balay $;
!
!  Include file for Fortran use of the TS (timestepping) package in PETSc
!
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
!
!  End of Fortran include file for the TS package in PETSc

