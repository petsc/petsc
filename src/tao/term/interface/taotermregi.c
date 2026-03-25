#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

PetscBool         TaoTermRegisterAllCalled = PETSC_FALSE;
PetscFunctionList TaoTermList              = NULL;

PETSC_INTERN PetscErrorCode TaoTermCreate_Callbacks(TaoTerm);
PETSC_INTERN PetscErrorCode TaoTermCreate_Shell(TaoTerm);
PETSC_INTERN PetscErrorCode TaoTermCreate_Sum(TaoTerm);
PETSC_INTERN PetscErrorCode TaoTermCreate_Halfl2squared(TaoTerm);
PETSC_INTERN PetscErrorCode TaoTermCreate_L1(TaoTerm);
PETSC_INTERN PetscErrorCode TaoTermCreate_Quadratic(TaoTerm);

/*@C
  TaoTermRegister - Register an implementation of `TaoTerm`

  Not Collective, No Fortran Support

  Input Parameters:
+ sname - name of a new user-defined term
- func  - routine to create the context for the `TaoTermType`

  Example Usage:
.vb
   TaoTermRegister("my_term", MyTermCreate);
.ve

  Then, your term can be chosen with the procedural interface via
$     TaoTermSetType(term, "my_term")
  or at runtime via the option
$     -tao_term_type my_term

  Level: advanced

  Note:
  `TaoTermRegister()` may be called multiple times to add multiple new `TaoTermType`.

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermSetType()`
@*/
PetscErrorCode TaoTermRegister(const char sname[], PetscErrorCode (*func)(TaoTerm))
{
  PetscFunctionBegin;
  PetscCall(TaoInitializePackage());
  PetscCall(PetscFunctionListAdd(&TaoTermList, sname, (void (*)(void))func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermRegisterAll(void)
{
  PetscFunctionBegin;
  if (TaoTermRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  TaoTermRegisterAllCalled = PETSC_TRUE;
  PetscCall(TaoTermRegister(TAOTERMCALLBACKS, TaoTermCreate_Callbacks));
  PetscCall(TaoTermRegister(TAOTERMSHELL, TaoTermCreate_Shell));
  PetscCall(TaoTermRegister(TAOTERMSUM, TaoTermCreate_Sum));
  PetscCall(TaoTermRegister(TAOTERMHALFL2SQUARED, TaoTermCreate_Halfl2squared));
  PetscCall(TaoTermRegister(TAOTERML1, TaoTermCreate_L1));
  PetscCall(TaoTermRegister(TAOTERMQUADRATIC, TaoTermCreate_Quadratic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
