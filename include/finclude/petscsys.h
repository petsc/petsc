!
!
!  Include file for Fortran use of the System package in PETSc
!
#if !defined (__PETSCSYS_H)
#define __PETSCSYS_H

#define PetscRandom PetscFortranAddr
#define PetscRandomType integer
#define PetscBinarySeekType integer

#endif


#if !defined (PETSC_AVOID_DECLARATIONS)
!
!     Random numbers
!
      integer RANDOM_DEFAULT,RANDOM_DEFAULT_REAL
      integer RANDOM_DEFAULT_IMAGINARY     

      parameter (RANDOM_DEFAULT=0,RANDOM_DEFAULT_REAL=1)
      parameter (RANDOM_DEFAULT_IMAGINARY=2)     
!
!
!
      integer PETSC_BINARY_INT_SIZE,PETSC_BINARY_FLOAT_SIZE
      integer PETSC_BINARY_CHAR_SIZE
      integer PETSC_BINARY_SHORT_SIZE,PETSC_BINARY_DOUBLE_SIZE
      integer PETSC_BINARY_SCALAR_SIZE

      parameter (PETSC_BINARY_INT_SIZE = 4)
      parameter (PETSC_BINARY_FLOAT_SIZE = 4)
      parameter (PETSC_BINARY_CHAR_SIZE = 1)
      parameter (PETSC_BINARY_SHORT_SIZE = 2)
      parameter (PETSC_BINARY_DOUBLE_SIZE = 8)
#if defined(PETSC_USE_COMPLEX)
      parameter (PETSC_BINARY_SCALAR_SIZE = 16)
#else
      parameter (PETSC_BINARY_SCALAR_SIZE = 8)
#endif

      integer PETSC_BINARY_SEEK_SET,PETSC_BINARY_SEEK_CUR
      integer PETSC_BINARY_SEEK_END

      parameter (PETSC_BINARY_SEEK_SET = 0,PETSC_BINARY_SEEK_CUR = 1)
      parameter (PETSC_BINARY_SEEK_END = 2)

!
!     End of Fortran include file for the System  package in PETSc

#endif
