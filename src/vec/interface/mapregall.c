#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mapregall.c,v 1.1 1999/06/21 02:03:51 knepley Exp $";
#endif

#include "vecimpl.h"     /*I  "vec.h"  I*/
EXTERN_C_BEGIN
extern int PetscMapCreate_MPI(PetscMap);
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
int PetscMapRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  PetscMapRegisterAllCalled = PETSC_TRUE;

  ierr = PetscMapRegisterDynamic(MAP_MPI, path, "PetscMapCreate_MPI", PetscMapCreate_MPI);                       CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

