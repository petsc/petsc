#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mapregall.c,v 1.1 1999/06/21 02:03:51 knepley Exp $";
#endif

#include "src/vec/vecimpl.h"     /*I  "vec.h"  I*/
EXTERN_C_BEGIN
extern int PetscMapCreate_MPI(PetscMap);

extern int PetscMapSerialize_MPI(MPI_Comm, PetscMap *, PetscViewer, PetscTruth);
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

  ierr = PetscMapRegister(MAP_MPI, path, "PetscMapCreate_MPI", PetscMapCreate_MPI);                       CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapSerializeRegisterAll"
/*@C
  PetscMapSerializeRegisterAll - Registers all of the serialization routines in the Vec package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: PetscMap, register, all, serialize
.seealso: PetscMapSerializeRegister(), PetscMapSerializeRegisterDestroy()
@*/
int PetscMapSerializeRegisterAll(const char path[])
{
  int ierr;

  PetscFunctionBegin;
  PetscMapSerializeRegisterAllCalled = PETSC_TRUE;

  ierr = PetscMapSerializeRegister(MAP_SER_MPI_BINARY, path, "PetscMapSerialize_MPI", PetscMapSerialize_MPI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
