!
!  $Id: snes.h,v 1.18 1999/02/04 23:07:06 bsmith Exp balay $;
!
!  Include file for Fortran use of the SNES package in PETSc
!
#define SNES            PetscFortranAddr
#define SNESProblemType integer
#define SNESType        character*(80)

!
!  SNESType
!
#define SNES_EQ_LS          'ls'
#define SNES_EQ_TR          'tr'
#define SNES_EQ_TR_DOG_LEG  
#define SNES_EQ_TR2_LIN
#define SNES_EQ_TEST        'test'
#define SNES_UM_LS          'umls'
#define SNES_UM_TR          'umtr'
#define SNES_LS_LM          'lslm'

!
!  Two classes of nonlinear solvers
!
      integer SNES_NONLINEAR_EQUATIONS
      integer SNES_UNCONSTRAINED_MINIMIZATION

      parameter (SNES_NONLINEAR_EQUATIONS = 0)
      parameter (SNES_UNCONSTRAINED_MINIMIZATION = 1)

!
!  Some PETSc fortran functions that the user might pass as arguments
!
      external SNESDEFAULTCOMPUTEJACOBIAN
      external SNESDEFAULTCOMPUTEJACOBIANWITHCOLORING
!
!  End of Fortran include file for the SNES package in PETSc




