!
!  $Id: ts.h,v 1.12 1999/02/04 23:07:23 bsmith Exp bsmith $;
!
!  Include file for Fortran use of the TS (timestepping) package in PETSc
!
#define TS            PetscFortranAddr
#define TSProblemType integer 
#define TSPVodeType   integer
#define TSType        character*(80)

#define TS_EULER  'euler'
#define TS_BEULER 'beuler'
#define TS_PSEUDO 'pseudo'
#define TS_PVODE  'pvode'

      integer TS_LINEAR, TS_NONLINEAR
      parameter (TS_LINEAR = 0, TS_NONLINEAR = 1)

      integer PVODE_ADAMS, PVODE_BDF
      parameter (PVODE_ADAMS=0, PVODE_BDF=1)
!
!  Some PETSc fortran functions that the user might pass as arguments
!
      external TSDEFAULTCOMPUTEJACOBIAN
      external TSDEFAULTCOMPUTEJACOBIANCOLOR
!
!  End of Fortran include file for the TS package in PETSc

