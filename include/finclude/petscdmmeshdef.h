!
!
!  Include file for Fortran use of the Mesh package in PETSc
!
#if !defined (__PETSCDMMESHDEF_H)
#define __PETSCDMMESHDEF_H

#include "finclude/petscdmdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define SectionReal PetscFortranAddr
#define SectionInt  PetscFortranAddr
#endif

#endif
