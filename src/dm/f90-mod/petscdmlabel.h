!
!  Used by petscdmmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscdmlabel.h"

      type tDMLabel
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tDMLabel

      DMLabel, parameter :: PETSC_NULL_DMLABEL = tDMLabel(0)

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_DMLABEL
#endif
