!
!
!  Include file for Fortran use of the IS (index set) package in PETSc
!
#if !defined (__PETSCIS_H)
#define __PETSCIS_H

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


#if !defined (PETSC_AVOID_DECLARATIONS)

#if defined(PETSC_USE_FORTRAN_MODULES) 
      type IS
        PetscFortranAddr:: v
      end type IS
      type ISColoring
        PetscFortranAddr:: v
      end type ISColoring
#endif

      PetscEnum IS_COLORING_GLOBAL
      PetscEnum IS_COLORING_GHOSTED
      parameter (IS_COLORING_GLOBAL = 0,IS_COLORING_GHOSTED = 1)

      PetscEnum IS_GENERAL
      PetscEnum IS_STRIDE
      PetscEnum IS_BLOCK
      parameter (IS_GENERAL = 0,IS_STRIDE = 1,IS_BLOCK = 2)

      PetscEnum IS_GTOLM_MASK
      PetscEnum IS_GTOLM_DROP 
      parameter (IS_GTOLM_MASK =0,IS_GTOLM_DROP = 1)

!
!  End of Fortran include file for the IS package in PETSc

#endif
