!
!
!  Include file for Fortran use of the Vec package in PETSc
!
#if !defined (__PETSCVEC_H)
#define __PETSCVEC_H

#if defined(PETSC_USE_FORTRAN_MODULES)
#define VEC_HIDE type(Vec)
#define VECSCATTER_HIDE type(VecScatter)
#define USE_VEC_HIDE use petscvecdef
#else
#define VEC_HIDE Vec
#define VECSCATTER_HIDE VecScatter
#define USE_VEC_HIDE

#define Vec PetscFortranAddr
#define VecScatter PetscFortranAddr
#endif

#define NormType PetscEnum
#define InsertMode PetscEnum
#define ScatterMode PetscEnum 
#define VecOption PetscEnum
#define VecType character*(80)
#define VecOperation PetscEnum

#define VECSEQ 'seq'
#define VECMPI 'mpi'
#define VECFETI 'feti'
#define VECSHARED 'shared'
#define VECESI 'esi'
#define VECPETSCESI 'petscesi'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)

#if defined(PETSC_USE_FORTRAN_MODULES)
      type Vec
        PetscFortranAddr:: v
      end type Vec
      type VecScatter
        PetscFortranAddr:: v
      end type VecScatter
#endif
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
      PetscEnum VEC_TREAT_OFF_PROC_ENTRIES
      PetscEnum VEC_IGNORE_NEGATIVE_INDICES
      PetscEnum VEC_TREAT_NEGATIVE_INDICES

      parameter (VEC_IGNORE_OFF_PROC_ENTRIES=0)
      parameter (VEC_TREAT_OFF_PROC_ENTRIES=1)
      parameter (VEC_IGNORE_NEGATIVE_INDICES=2)
      parameter (VEC_TREAT_NEGATIVE_INDICES=3)

!
!  VecOperation
!
      PetscEnum VECOP_VIEW
      PetscEnum VECOP_LOADINTOVECTOR

      parameter (VECOP_VIEW=34,VECOP_LOADINTOVECTOR=41)
!
!  End of Fortran include file for the Vec package in PETSc

#endif
