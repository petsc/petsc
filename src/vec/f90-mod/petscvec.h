!
!  Used by petscvecmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscvec.h"

      type tVec
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tVec
      type tVecScatter
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tVecScatter
      type tVecTagger
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tVecTagger

      Vec, parameter :: PETSC_NULL_VEC = tVec(0)
      VecScatter, parameter :: PETSC_NULL_VECSCATTER = tVecScatter(0)
      VecTagger, parameter :: PETSC_NULL_VECTAGGER = tVecTagger(0)
!
!
!  Types of vector and matrix norms
!
      PetscEnum, parameter :: NORM_1 = 0
      PetscEnum, parameter :: NORM_2 = 1
      PetscEnum, parameter :: NORM_FROBENIUS = 2
      PetscEnum, parameter :: NORM_INFINITY = 3
      PetscEnum, parameter :: NORM_MAX = 3
      PetscEnum, parameter :: NORM_1_AND_2 = 4
!
!  Flags for VecSetValues() and MatSetValues()
!
      PetscEnum, parameter :: NOT_SET_VALUES = 0
      PetscEnum, parameter :: INSERT_VALUES = 1
      PetscEnum, parameter :: ADD_VALUES = 2
      PetscEnum, parameter :: MAX_VALUES = 3
      PetscEnum, parameter :: MIN_VALUES = 4
      PetscEnum, parameter :: INSERT_ALL_VALUES = 5
      PetscEnum, parameter :: ADD_ALL_VALUES = 6
      PetscEnum, parameter :: INSERT_BC_VALUES = 7
      PetscEnum, parameter :: ADD_BC_VALUES = 8
!
!  Types of vector scatters
!
      PetscEnum, parameter :: SCATTER_FORWARD = 0
      PetscEnum, parameter :: SCATTER_REVERSE = 1
      PetscEnum, parameter :: SCATTER_FORWARD_LOCAL = 2
      PetscEnum, parameter :: SCATTER_REVERSE_LOCAL = 3
!
!  VecOption
!
      PetscEnum, parameter :: VEC_IGNORE_OFF_PROC_ENTRIES = 0
      PetscEnum, parameter :: VEC_IGNORE_NEGATIVE_INDICES = 1
      PetscEnum, parameter :: VEC_SUBSET_OFF_PROC_ENTRIES = 2
!
!  VecOperation
!
      PetscEnum, parameter :: VECOP_DUPLICATE = 0
      PetscEnum, parameter :: VECOP_VIEW = 33
      PetscEnum, parameter :: VECOP_LOAD = 41
      PetscEnum, parameter :: VECOP_VIEWNATIVE = 68
      PetscEnum, parameter :: VECOP_LOADNATIVE = 69

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_VEC
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_VECSCATTER
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_VECTAGGER
!DEC$ ATTRIBUTES DLLEXPORT::NORM_1
!DEC$ ATTRIBUTES DLLEXPORT::NORM_2
!DEC$ ATTRIBUTES DLLEXPORT::NORM_FROBENIUS
!DEC$ ATTRIBUTES DLLEXPORT::NORM_INFINITY
!DEC$ ATTRIBUTES DLLEXPORT::NORM_MAX
!DEC$ ATTRIBUTES DLLEXPORT::NORM_1_AND_2
!DEC$ ATTRIBUTES DLLEXPORT::NOT_SET_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::INSERT_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::ADD_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::MAX_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::INSERT_ALL_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::ADD_ALL_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::INSERT_BC_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::ADD_BC_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::SCATTER_FORWARD
!DEC$ ATTRIBUTES DLLEXPORT::SCATTER_REVERSE
!DEC$ ATTRIBUTES DLLEXPORT::SCATTER_FORWARD_LOCAL
!DEC$ ATTRIBUTES DLLEXPORT::SCATTER_REVERSE_LOCAL
!DEC$ ATTRIBUTES DLLEXPORT::VEC_IGNORE_OFF_PROC_ENTRIES
!DEC$ ATTRIBUTES DLLEXPORT::VEC_IGNORE_NEGATIVE_INDICES
!DEC$ ATTRIBUTES DLLEXPORT::VEC_SUBSET_OFF_PROC_ENTRIES
!DEC$ ATTRIBUTES DLLEXPORT::VECOP_DUPLICATE
!DEC$ ATTRIBUTES DLLEXPORT::VECOP_VIEW
!DEC$ ATTRIBUTES DLLEXPORT::VECOP_LOAD
!DEC$ ATTRIBUTES DLLEXPORT::VECOP_VIEWNATIVE
!DEC$ ATTRIBUTES DLLEXPORT::VECOP_LOADNATIVE
#endif
