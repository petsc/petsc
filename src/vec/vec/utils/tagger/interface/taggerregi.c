#include <petsc/private/vecimpl.h> /*I  "petscvec.h"  I*/

PETSC_INTERN PetscErrorCode VecTaggerCreate_Absolute(VecTagger);
PETSC_INTERN PetscErrorCode VecTaggerCreate_Relative(VecTagger);
PETSC_INTERN PetscErrorCode VecTaggerCreate_CDF(VecTagger);
PETSC_INTERN PetscErrorCode VecTaggerCreate_Or(VecTagger);
PETSC_INTERN PetscErrorCode VecTaggerCreate_And(VecTagger);

PetscFunctionList VecTaggerList;

/*@C
  VecTaggerRegisterAll - Registers all the `VecTagger` communication implementations

  Not Collective

  Level: advanced

.seealso: `VecTaggerRegisterDestroy()`
@*/
PetscErrorCode VecTaggerRegisterAll(void)
{
  PetscFunctionBegin;
  if (VecTaggerRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  VecTaggerRegisterAllCalled = PETSC_TRUE;
  PetscCall(VecTaggerRegister(VECTAGGERABSOLUTE, VecTaggerCreate_Absolute));
  PetscCall(VecTaggerRegister(VECTAGGERRELATIVE, VecTaggerCreate_Relative));
  PetscCall(VecTaggerRegister(VECTAGGERCDF, VecTaggerCreate_CDF));
  PetscCall(VecTaggerRegister(VECTAGGEROR, VecTaggerCreate_Or));
  PetscCall(VecTaggerRegister(VECTAGGERAND, VecTaggerCreate_And));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  VecTaggerRegister  - Adds an implementation of the `VecTagger` communication protocol.

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - name of a new user-defined implementation
- function - routine to create method context

  Level: advanced

  Notes:
  `VecTaggerRegister()` may be called multiple times to add several user-defined implementations.

  Example Usage:
.vb
   VecTaggerRegister("my_impl", MyImplCreate);
.ve

  Then, this implementation can be chosen with the procedural interface via
.vb
  VecTaggerSetType(tagger, "my_impl")
.ve
  or at runtime via the option
.vb
  -snes_type my_solver
.ve

.seealso: `VecTaggerType`, `VecTaggerCreate()`, `VecTagger`, `VecTaggerRegisterAll()`, `VecTaggerRegisterDestroy()`
@*/
PetscErrorCode VecTaggerRegister(const char sname[], PetscErrorCode (*function)(VecTagger))
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListAdd(&VecTaggerList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}
