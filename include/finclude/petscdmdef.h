
!
!  Include file for Fortran use of the DM package in PETSc
!
#if !defined (__PETSCDMDEF_H)
#define __PETSCDMDEF_H

#include "finclude/petscisdef.h"
#include "finclude/petscvecdef.h"
#include "finclude/petscmatdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define DM PetscFortranAddr
#endif

#endif
