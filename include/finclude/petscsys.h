C
C  $Id: sys.h,v 1.4 1996/04/16 04:58:49 bsmith Exp balay $;
C
C  Include file for Fortran use of the System package in PETSc
C
#define PetscRandom     integer
#define PetscBinaryType integer
#define PetscRandomType integer
C
C     Random numbers
C
      integer   RANDOM_DEFAULT, RANDOM_DEFAULT_REAL,
     *          RANDOM_DEFAULT_IMAGINARY     
      parameter (RANDOM_DEFAULT=0, RANDOM_DEFAULT_REAL=1,
     *           RANDOM_DEFAULT_IMAGINARY=2)     
C
C Not used from Fortran 
C
      integer BINARY_INT, BINARY_DOUBLE,BINARY_SCALAR, BINARY_SHORT,
     *        BINARY_FLOAT,BINARY_CHAR
      parameter (BINARY_INT=0, BINARY_DOUBLE=1,BINARY_SCALAR=1,
     *           BINARY_SHORT=2,BINARY_FLOAT=3,BINARY_CHAR=4)

C
C     End of Fortran include file for the System  package in PETSc
