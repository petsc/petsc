#include <petsc/private/taoimpl.h>

PETSC_INTERN PetscErrorCode TaoTermCreate_ElementwiseDivergence_Internal(TaoTerm term)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)term->solution_factory));
  PetscCall(MatDestroy(&term->parameters_factory));
  term->parameters_factory = term->solution_factory;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TaoTermDestroy_ElementwiseDivergence_Internal(TaoTerm term)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy(&term->parameters_factory));
  PetscCall(PetscObjectReference((PetscObject)term->parameters_factory_orig));
  term->parameters_factory = term->parameters_factory_orig;
  PetscFunctionReturn(PETSC_SUCCESS);
}
