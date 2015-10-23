!
!
!  Include file for Fortran use of the IS (index set) package in PETSc
!
#if !defined (__PETSCISDEF_H)
#define __PETSCISDEF_H

#include "petsc/finclude/petscsysdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define IS PetscFortranAddr
#define ISColoring PetscFortranAddr
#define PetscSection PetscFortranAddr
#endif

#define PetscSF PetscFortranAddr
#define PetscLayout PetscFortranAddr

#define ISType PetscEnum
#define ISLocalToGlobalMapping PetscFortranAddr
#define ISGlobalToLocalMappingType PetscEnum
#define ISColoringType PetscEnum

#if PETSC_IS_COLOR_VALUE_TYPE_SIZE == 1
#define ISColoringValue integer1
#elif PETSC_IS_COLOR_VALUE_TYPE_SIZE == 2
#define ISColoringValue integer2
#else
#error "Unknown size for IS_COLOR_VALUE_TYPE"
#endif

#define ISGENERAL 'general'
#define ISSTRIDE 'stride'
#define ISBLOCK 'block'
#endif
