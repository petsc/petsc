!
!
!  Include file for Fortran use of the IS (index set) package in PETSc
!
#if !defined (__PETSCIS_H)
#define __PETSCIS_H

#define IS PetscFortranAddr
#define ISType PetscEnum
#define ISColoring PetscFortranAddr
#define ISLocalToGlobalMapping PetscFortranAddr
#define ISGlobalToLocalMappingType PetscEnum
#define ISColoringType PetscEnum


#endif


#if !defined (PETSC_AVOID_DECLARATIONS)

      PetscEnum IS_COLORING_GLOBAL,IS_COLORING_GHOSTED
      parameter (IS_COLORING_GLOBAL = 0,IS_COLORING_GHOSTED = 1)

      PetscEnum IS_GENERAL,IS_STRIDE,IS_BLOCK
      parameter (IS_GENERAL = 0,IS_STRIDE = 1,IS_BLOCK = 2)

      PetscEnum IS_GTOLM_MASK,IS_GTOLM_DROP 
      parameter (IS_GTOLM_MASK =0,IS_GTOLM_DROP = 1)

!
!  End of Fortran include file for the IS package in PETSc

#endif
