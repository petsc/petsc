!
!  Used by petscvecmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscao.h"

!  cannot use tAO because that type matches the variable tao used in tao examples
      type, extends(tPetscObject) :: tPetscAO
      end type tPetscAO
      AO, parameter :: PETSC_NULL_AO = tPetscAO(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_AO
#endif
