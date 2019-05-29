#include <petsc/private/sfimpl.h>     /*I  "petscsf.h"  I*/

PETSC_EXTERN PetscErrorCode PetscSFCreate_Basic(PetscSF);
#if defined(PETSC_HAVE_MPI_WIN_CREATE) && defined(PETSC_HAVE_MPI_TYPE_DUP)
PETSC_EXTERN PetscErrorCode PetscSFCreate_Window(PetscSF);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscSFRegisterAllCalled) PetscFunctionReturn(0);
  PetscSFRegisterAllCalled = PETSC_TRUE;
  ierr = PetscSFRegister(PETSCSFBASIC,  PetscSFCreate_Basic);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_WIN_CREATE) && defined(PETSC_HAVE_MPI_TYPE_DUP)
  ierr = PetscSFRegister(PETSCSFWINDOW, PetscSFCreate_Window);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscSFList,name,create);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
