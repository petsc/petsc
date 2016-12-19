#include <petsc/private/sfimpl.h>     /*I  "petscsf.h"  I*/

#if defined(PETSC_HAVE_MPI_WIN_CREATE)
PETSC_EXTERN PetscErrorCode PetscSFCreate_Window(PetscSF);
#endif
PETSC_EXTERN PetscErrorCode PetscSFCreate_Basic(PetscSF);

PetscFunctionList PetscSFList;

/*@C
   PetscSFRegisterAll - Registers all the PetscSF communication implementations

   Not Collective

   Level: advanced

.keywords: PetscSF, register, all

.seealso:  PetscSFRegisterDestroy()
@*/
PetscErrorCode  PetscSFRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscSFRegisterAllCalled) PetscFunctionReturn(0);
  PetscSFRegisterAllCalled = PETSC_TRUE;
#if defined(PETSC_HAVE_MPI_WIN_CREATE) && defined(PETSC_HAVE_MPI_TYPE_DUP)
  ierr = PetscSFRegister(PETSCSFWINDOW, PetscSFCreate_Window);CHKERRQ(ierr);
#endif
  ierr = PetscSFRegister(PETSCSFBASIC,  PetscSFCreate_Basic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSFRegister  - Adds an implementation of the PetscSF communication protocol.

   Not collective

   Input Parameters:
+  name_impl - name of a new user-defined implementation
-  routine_create - routine to create method context

   Notes:
   PetscSFRegister() may be called multiple times to add several user-defined implementations.

   Sample usage:
.vb
   PetscSFRegister("my_impl",MyImplCreate);
.ve

   Then, this implementation can be chosen with the procedural interface via
$     PetscSFSetType(sf,"my_impl")
   or at runtime via the option
$     -snes_type my_solver

   Level: advanced

.keywords: PetscSF, register

.seealso: PetscSFRegisterAll(), PetscSFRegisterDestroy()
@*/
PetscErrorCode  PetscSFRegister(const char sname[],PetscErrorCode (*function)(PetscSF))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscSFList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

