!
!  Used by petscdmmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscdmlabel.h"

      type tDMLabel
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tDMLabel

      DMLabel, parameter :: PETSC_NULL_DMLABEL = tDMLabel(0)
