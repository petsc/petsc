C
C  $Id: snes.h,v 1.12 1997/11/13 19:38:17 balay Exp bsmith $;
C
C  Include file for Fortran use of the SNES package in PETSc
C
#define SNES            integer
#define SNESProblemType integer

C
C  SNESType
C
#define SNES_EQ_LS          'ls'
#define SNES_EQ_TR          'tr'
#define SNES_EQ_TR_DOG_LEG  
#define SNES_EQ_TR2_LIN
#define SNES_EQ_TEST        'test'
#define SNES_UM_LS          'umls'
#define SNES_UM_TR          'umtr'
#define SNES_LS_LM          'lslm'

C
C  Two classes of nonlinear solvers
C
      integer SNES_NONLINEAR_EQUATIONS,
     *        SNES_UNCONSTRAINED_MINIMIZATION

      parameter (SNES_NONLINEAR_EQUATIONS = 0,
     *           SNES_UNCONSTRAINED_MINIMIZATION = 1)

C
C  End of Fortran include file for the SNES package in PETSc




