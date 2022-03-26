#include <petsc/private/sfimpl.h>     /*I  "petscsf.h"  I*/

PETSC_INTERN PetscErrorCode PetscSFCreate_Basic(PetscSF);
#if defined(PETSC_HAVE_MPI_WIN_CREATE)
PETSC_INTERN PetscErrorCode PetscSFCreate_Window(PetscSF);
#endif
PETSC_INTERN PetscErrorCode PetscSFCreate_Allgatherv(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFCreate_Allgather(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFCreate_Gatherv(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFCreate_Gather(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFCreate_Alltoall(PetscSF);
#if defined(PETSC_HAVE_MPI_NEIGHBORHOOD_COLLECTIVES)
PETSC_INTERN PetscErrorCode PetscSFCreate_Neighbor(PetscSF);
#endif

PetscFunctionList PetscSFList;
PetscBool         PetscSFRegisterAllCalled;

/*@C
   PetscSFRegisterAll - Registers all the PetscSF communication implementations

   Not Collective

   Level: advanced

.seealso:  PetscSFRegisterDestroy()
@*/
PetscErrorCode  PetscSFRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscSFRegisterAllCalled) PetscFunctionReturn(0);
  PetscSFRegisterAllCalled = PETSC_TRUE;
  PetscCall(PetscSFRegister(PETSCSFBASIC,  PetscSFCreate_Basic));
#if defined(PETSC_HAVE_MPI_WIN_CREATE)
  PetscCall(PetscSFRegister(PETSCSFWINDOW, PetscSFCreate_Window));
#endif
  PetscCall(PetscSFRegister(PETSCSFALLGATHERV,PetscSFCreate_Allgatherv));
  PetscCall(PetscSFRegister(PETSCSFALLGATHER, PetscSFCreate_Allgather));
  PetscCall(PetscSFRegister(PETSCSFGATHERV,   PetscSFCreate_Gatherv));
  PetscCall(PetscSFRegister(PETSCSFGATHER,    PetscSFCreate_Gather));
  PetscCall(PetscSFRegister(PETSCSFALLTOALL,  PetscSFCreate_Alltoall));
#if defined(PETSC_HAVE_MPI_NEIGHBORHOOD_COLLECTIVES)
  PetscCall(PetscSFRegister(PETSCSFNEIGHBOR,  PetscSFCreate_Neighbor));
#endif
  PetscFunctionReturn(0);
}

/*@C
  PetscSFRegister  - Adds an implementation of the PetscSF communication protocol.

   Not collective

   Input Parameters:
+  name - name of a new user-defined implementation
-  create - routine to create method context

   Notes:
   PetscSFRegister() may be called multiple times to add several user-defined implementations.

   Sample usage:
.vb
   PetscSFRegister("my_impl",MyImplCreate);
.ve

   Then, this implementation can be chosen with the procedural interface via
$     PetscSFSetType(sf,"my_impl")
   or at runtime via the option
$     -sf_type my_impl

   Level: advanced

.seealso: PetscSFRegisterAll(), PetscSFInitializePackage()
@*/
PetscErrorCode  PetscSFRegister(const char name[],PetscErrorCode (*create)(PetscSF))
{
  PetscFunctionBegin;
  PetscCall(PetscSFInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscSFList,name,create));
  PetscFunctionReturn(0);
}
