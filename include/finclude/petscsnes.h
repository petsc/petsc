C
C  $Id: snes.h,v 1.10 1996/02/12 20:29:04 bsmith Exp balay $;
C
C  Include file for Fortran use of the SNES package in PETSc
C
#define SNES            integer
#define SNESProblemType integer
#define SNESType        integer

C
C  SNESType
C
      integer SNES_UNKNOWN_METHOD,SNES_EQ_LS,SNES_EQ_TR,
     *        SNES_EQ_TR_DOG_LEG,SNES_EQ_TR2_LIN,SNES_EQ_TEST,
     *        SNES_UM_LS,SNES_UM_TR 

      parameter (SNES_UNKNOWN_METHOD=-1,SNES_EQ_LS=0,SNES_EQ_TR=1,
     *           SNES_EQ_TR_DOG_LEG=2,SNES_EQ_TR2_LIN=3,SNES_EQ_TEST=4,
     *           SNES_UM_LS=5,SNES_UM_TR=6)

C
C  Two classes of nonlinear solvers
C
      integer SNES_NONLINEAR_EQUATIONS,
     *        SNES_UNCONSTRAINED_MINIMIZATION

      parameter (SNES_NONLINEAR_EQUATIONS = 0,
     *           SNES_UNCONSTRAINED_MINIMIZATION = 1)

C
C  End of Fortran include file for the SNES package in PETSc




