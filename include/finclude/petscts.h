!
!  $Id: ts.h,v 1.14 1999/03/24 18:11:29 balay Exp balay $;
!
!  Include file for Fortran use of the TS (timestepping) package in PETSc
!
#if !defined (__TS_H)
#define __TS_H

#define TS            PetscFortranAddr
#define TSProblemType integer 
#define TSPVodeType   integer
#define TSType        character*(80)

#define TS_EULER  'euler'
#define TS_BEULER 'beuler'
#define TS_PSEUDO 'pseudo'
#define TS_PVODE  'pvode'

#endif

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

