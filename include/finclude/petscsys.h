C
C  $Id: sys.h,v 1.7 1997/09/18 14:40:56 bsmith Exp bsmith $;
C
C  Include file for Fortran use of the System package in PETSc
C
#define PetscRandom     integer
#define PetscDataType   integer
#define PetscRandomType integer
C
C     Random numbers
C
      integer   RANDOM_DEFAULT, RANDOM_DEFAULT_REAL,
     *          RANDOM_DEFAULT_IMAGINARY     
      parameter (RANDOM_DEFAULT=0, RANDOM_DEFAULT_REAL=1,
     *           RANDOM_DEFAULT_IMAGINARY=2)     
C
C
C
      integer BINARY_INT_SIZE, BINARY_FLOAT_SIZE, BINARY_CHAR_SIZE,
     *        BINARY_SHORT_SIZE, BINARY_DOUBLE_SIZE, 
     *        BINARY_SCALAR_SIZE

      parameter (BINARY_INT_SIZE = 32, BINARY_FLOAT_SIZE = 32,
     *            BINARY_CHAR_SIZE = 8, BINARY_SHORT_SIZE = 16,
     *            BINARY_DOUBLE_SIZE = 64)
#if defined(PETSC_COMPLEX)
      parameter ( BINARY_SCALAR_SIZE = 128)
#else
      parameter ( BINARY_SCALAR_SIZE = 64)
#endif

      integer BINARY_SEEK_SET, BINARY_SEEK_CUR, BINARY_SEEK_END
      parameter (BINARY_SEEK_SET = 0, BINARY_SEEK_CUR = 1,
     *            BINARY_SEEK_END = 2)

C
C     End of Fortran include file for the System  package in PETSc
