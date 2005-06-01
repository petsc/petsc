#define PETSCVEC_DLL

#include "private/vecimpl.h"     /*I  "vec.h"  I*/
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT PetscMapCreate_MPI(PetscMap);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscMapRegisterAll"
/*@C
  PetscMapRegisterAll - Registers all of the map components in the Vec package. 

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: map, register, all
.seealso: PetscMapRegister(), PetscMapRegisterDestroy()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscMapRegisterAllCalled = PETSC_TRUE;

  ierr = PetscMapRegisterDynamic(MAP_MPI, path, "PetscMapCreate_MPI", PetscMapCreate_MPI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

