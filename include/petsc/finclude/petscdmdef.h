
!
!  Include file for Fortran use of the DM package in PETSc
!
#if !defined (__PETSCDMDEF_H)
#define __PETSCDMDEF_H

#include "petsc/finclude/petscisdef.h"
#include "petsc/finclude/petscvecdef.h"
#include "petsc/finclude/petscmatdef.h"

#define DMBoundaryType PetscEnum

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define DM               PetscFortranAddr
#define PetscQuadrature  PetscFortranAddr
#define PetscDS          PetscFortranAddr
#define PetscFE          PetscFortranAddr
#define PetscSpace       PetscFortranAddr
#define PetscDualSpace   PetscFortranAddr
#define PetscFV          PetscFortranAddr
#define PetscLimiter     PetscFortranAddr
#define PetscPartitioner PetscFortranAddr
#endif

#endif
