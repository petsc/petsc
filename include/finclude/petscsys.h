C
C  $Id: mat.h,v 1.9 1996/02/12 20:30:32 bsmith Exp $;
C
C  Include file for Fortran use of the System package in PETSc
C

C
C     Random numbers
C
      integer   RANDOM_DEFAULT, RANDOM_DEFAULT_REAL,
     *          RANDOM_DEFAULT_IMAGINARY     

      parameter (RANDOM_DEFAULT=0, RANDOM_DEFAULT_REAL=1,
     *           RANDOM_DEFAULT_IMAGINARY=2)     
C
C     Random number object
#define PetscRandom integer
C



C     End of Fortran include file for the System  package in PETSc
