!
!
!  Include file for Fortran use of the Vec package in PETSc
!
#include "petsc/finclude/petscvec.h"

      type tVec
        sequence
        PetscFortranAddr:: v
      end type tVec
      type tVecScatter
        sequence
        PetscFortranAddr:: v
      end type tVecScatter
      type tVecTagger
        sequence
        PetscFortranAddr:: v
      end type tVecTagger

      Vec, parameter :: PETSC_NULL_VEC = tVec(-1)
      VecScatter, parameter :: PETSC_NULL_VECSCATTER =                    &
     &                tVecScatter(-1)
      VecTagger, parameter :: PETSC_NULL_VECTAGGER =                      &
     &                tVecTagger(-1)
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
      PetscEnum INSERT_ALL_VALUES
      PetscEnum ADD_ALL_VALUES
      PetscEnum INSERT_BC_VALUES
      PetscEnum ADD_BC_VALUES

      parameter (NOT_SET_VALUES=0,INSERT_VALUES=1,ADD_VALUES=2)
      parameter (MAX_VALUES=3)
      parameter (INSERT_ALL_VALUES=4,ADD_ALL_VALUES=5)
      parameter (INSERT_BC_VALUES=6,ADD_BC_VALUES=7)
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
      PetscEnum VEC_SUBSET_OFF_PROC_ENTRIES

      parameter (VEC_IGNORE_OFF_PROC_ENTRIES=0)
      parameter (VEC_IGNORE_NEGATIVE_INDICES=1)
      parameter (VEC_SUBSET_OFF_PROC_ENTRIES=2)

!
!  VecOperation
!
      PetscEnum VECOP_DUPLICATE
      PetscEnum VECOP_VIEW
      PetscEnum VECOP_LOAD
      PetscEnum VECOP_VIEWNATIVE
      PetscEnum VECOP_LOADNATIVE

      parameter (VECOP_DUPLICATE=0,VECOP_VIEW=33,VECOP_LOAD=41)
      parameter (VECOP_VIEWNATIVE=68,VECOP_LOADNATIVE=69)
!
!  End of Fortran include file for the Vec package in PETSc

