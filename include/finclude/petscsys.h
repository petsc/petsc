!
!  $Id: sys.h,v 1.18 1999/09/14 15:58:43 balay Exp bsmith $;
!
!  Include file for Fortran use of the System package in PETSc
!
#if !defined (__SYS_H)
#define __SYS_H

#define PetscRandom         PetscFortranAddr
#define PetscBinarySeekType integer
#define PetscRandomType     integer

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
      integer BINARY_INT_SIZE,BINARY_FLOAT_SIZE,BINARY_CHAR_SIZE
      integer BINARY_SHORT_SIZE,BINARY_DOUBLE_SIZE
      integer BINARY_SCALAR_SIZE

      parameter (BINARY_INT_SIZE = 32,BINARY_FLOAT_SIZE = 32)
      parameter (BINARY_CHAR_SIZE = 8,BINARY_SHORT_SIZE = 16)
      parameter (BINARY_DOUBLE_SIZE = 64)
#if defined(PETSC_USE_COMPLEX)
      parameter (BINARY_SCALAR_SIZE = 128)
#else
      parameter (BINARY_SCALAR_SIZE = 64)
#endif

      integer BINARY_SEEK_SET,BINARY_SEEK_CUR,BINARY_SEEK_END

      parameter (BINARY_SEEK_SET = 0,BINARY_SEEK_CUR = 1)
      parameter (BINARY_SEEK_END = 2)

!
!     End of Fortran include file for the System  package in PETSc

#endif
