!
!  Part of the base include file for Fortran use of PETSc IS
!  Note: This file should contain only define statements

! No spaces for #defines as some compilers (PGI) also adds
! those additional spaces during preprocessing - bad for fixed format
!
#if !defined (PETSCISBASEDEF_H)
#define PETSCISBASEDEF_H

#define ISColoringValue PETSC_IS_COLORING_VALUE_TYPE_F

#endif
