!
!
!  Include file for Fortran use of the Vec package in PETSc
!
#include "finclude/petscvecdef.h"
!
!
!  Types of vector and matrix norms
!
      PetscEnum NORM_1
      PetscEnum NORM_2
      PetscEnum NORM_FROBENIUS
      PetscEnum NORM_INFINITY
      PetscEnum NORM_MAX
      PetscEnum NORM_1_AND_2

      parameter (NORM_1=0,NORM_2=1,NORM_FROBENIUS=2,NORM_INFINITY=3)
      parameter (NORM_MAX=3,NORM_1_AND_2=4)
!
!  Flags for VecSetValues() and MatSetValues()
!
      PetscEnum NOT_SET_VALUES
      PetscEnum INSERT_VALUES
      PetscEnum ADD_VALUES
      PetscEnum MAX_VALUES

      parameter (NOT_SET_VALUES=0,INSERT_VALUES=1,ADD_VALUES=2)
      parameter (MAX_VALUES=3)
!
!  Types of vector scatters
!
      PetscEnum SCATTER_FORWARD
      PetscEnum SCATTER_REVERSE
      PetscEnum SCATTER_FORWARD_LOCAL
      PetscEnum SCATTER_REVERSE_LOCAL
      PetscEnum SCATTER_LOCAL

      parameter (SCATTER_FORWARD=0,SCATTER_REVERSE=1)
      parameter (SCATTER_FORWARD_LOCAL=2,SCATTER_REVERSE_LOCAL=3)
      parameter (SCATTER_LOCAL=2)
!
!  VecOption
!
      PetscEnum VEC_IGNORE_OFF_PROC_ENTRIES
      PetscEnum VEC_IGNORE_NEGATIVE_INDICES

      parameter (VEC_IGNORE_OFF_PROC_ENTRIES=0)
      parameter (VEC_IGNORE_NEGATIVE_INDICES=1)

!
!  VecOperation
!
      PetscEnum VECOP_VIEW
      PetscEnum VECOP_LOADINTOVECTOR

      parameter (VECOP_VIEW=33,VECOP_LOADINTOVECTOR=41)
!
!  End of Fortran include file for the Vec package in PETSc

