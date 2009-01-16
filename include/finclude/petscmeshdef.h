!
!
!  Include file for Fortran use of the Mesh package in PETSc
!
#if !defined (__PETSCMESHDEF_H)
#define __PETSCMESHDEF_H

#include "finclude/petscdadef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define Mesh PetscFortranAddr
#define SectionReal PetscFortranAddr
#define SectionInt  PetscFortranAddr
#endif

#define MeshType character*(80)

#define MESHSIEVE 'sieve'

#endif
