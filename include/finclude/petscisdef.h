!
!
!  Include file for Fortran use of the IS (index set) package in PETSc
!
#if !defined (__PETSCISDEF_H)
#define __PETSCISDEF_H

#if defined(PETSC_USE_FORTRAN_MODULES) 
#define IS_HIDE type(IS)
#define ISCOLORING_HIDE type(ISColoring)
#define USE_IS_HIDE use petscisdef
#else
#define IS_HIDE IS
#define ISCOLORING_HIDE ISColoring
#define USE_IS_HIDE

#define IS PetscFortranAddr
#define ISColoring PetscFortranAddr
#endif

#define ISType PetscEnum
#define ISLocalToGlobalMapping PetscFortranAddr
#define ISGlobalToLocalMappingType PetscEnum
#define ISColoringType PetscEnum


#endif
