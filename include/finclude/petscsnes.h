!
!  $Id: snes.h,v 1.22 1999/04/01 19:55:13 balay Exp balay $;
!
!  Include file for Fortran use of the SNES package in PETSc
!
#if !defined (__SNES_H)
#define __SNES_H

#define SNES            PetscFortranAddr
#define SNESProblemType integer
#define SNESType        character*(80)
!
!  SNESType
!
#define SNES_EQ_LS          'ls'
#define SNES_EQ_TR          'tr'
#define SNES_EQ_TEST        'test'
#define SNES_UM_LS          'umls'
#define SNES_UM_TR          'umtr'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)
!
!  Two classes of nonlinear solvers
!
      integer SNES_NONLINEAR_EQUATIONS
      integer SNES_UNCONSTRAINED_MINIMIZATION
      integer SNES_LEAST_SQUARES

      parameter (SNES_NONLINEAR_EQUATIONS = 0)
      parameter (SNES_UNCONSTRAINED_MINIMIZATION = 1)
      parameter (SNES_LEAST_SQUARES = 2)
!
!  Some PETSc fortran functions that the user might pass as arguments
!
      external SNESDEFAULTCOMPUTEJACOBIAN
      external SNESDEFAULTCOMPUTEJACOBIANCOLOR
!
!  End of Fortran include file for the SNES package in PETSc

#endif
