C
C  Include file for Fortran use of the SNES package in PETSc
C
#define SNES            integer
#define SNESProblemType integer

C
C  Two classes of nonlinear solvers
C
      integer SNES_NONLINEAR_EQUATIONS,
     *        SNES_UNCONSTRAINED_MINIMIZATION

      parameter (SNES_NONLINEAR_EQUATIONS = 0,
     *           SNES_UNCONSTRAINED_MINIMIZATION = 1)

C
C  End of Fortran include file for the SNES package in PETSc

