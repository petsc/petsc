C
C      Include file for Fortran use of the SNES package in PETSc
C
#define SNES     integer
#define SNESType integer

      integer SNES_NONLINEAR_EQUATIONS,
     *        SNES_UNCONSTRAINED_MINIMIZATION

      parameter (SNES_NONLINEAR_EQUATIONS = 0,
     *           SNES_UNCONSTRAINED_MINIMIZATION = 1)

      integer POSITIVE_FUNCTION_VALUE, NEGATIVE_FUNCTION_VALUE

      parameter (POSITIVE_FUNCTION_VALUE = 0, 
     *           NEGATIVE_FUNCTION_VALUE = 1)
C
C      End of Fortran include file for the SNES package in PETSc

