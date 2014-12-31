!
!  Note: MAT_FACTORINFO_SIZE must equal # elements in MatFactorInfo structure
!  (See petsc/include/petscmat.h)
! Note: This is needed in f90 interface for MatGetInfo() - hence
! in a separate include
!
      PetscEnum MAT_FACTORINFO_SIZE
      parameter (MAT_FACTORINFO_SIZE=11)
