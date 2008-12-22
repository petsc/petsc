!
!
!  Include file for Fortran use of the Mesh package in PETSc
!
#if !defined (__PETSCMESHDEF_H)
#define __PETSCMESHDEF_H

#include "finclude/petscdadef.h"

#define Mesh PetscFortranAddr
#define MeshType character*(80)

#define MESHSIEVE 'sieve'

#define SectionReal PetscFortranAddr
#define SectionInt  PetscFortranAddr

#endif
