#include <petsc/private/ftnimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define sneslinesearchsetprecheck_  SNESLINESEARCHSETPRECHECK
  #define sneslinesearchgetprecheck_  SNESLINESEARCHGETPRECHECK
  #define sneslinesearchsetpostcheck_ SNESLINESEARCHSETPOSTCHECK
  #define sneslinesearchgetpostcheck_ SNESLINESEARCHGETPOSTCHECK
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define sneslinesearchsetprecheck_  sneslinesearchsetprecheck
  #define sneslinesearchgetprecheck_  sneslinesearchgetprecheck
  #define sneslinesearchsetpostcheck_ sneslinesearchsetpostcheck
  #define sneslinesearchgetpostcheck_ sneslinesearchgetpostcheck
#endif

/* fortranpointers go: shell, precheck, postcheck */

static PetscErrorCode oursneslinesearchprecheck(SNESLineSearch linesearch, Vec X, Vec Y, PetscBool *changed, void *ctx)
{
  PetscFunctionBegin;
  PetscCallFortranVoidFunction((*(void (*)(SNESLineSearch *, Vec *, Vec *, PetscBool *, void *, PetscErrorCode *))(((PetscObject)linesearch)->fortran_func_pointers[1]))(&linesearch, &X, &Y, changed, ctx, &ierr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode oursneslinesearchpostcheck(SNESLineSearch linesearch, Vec X, Vec Y, Vec W, PetscBool *changed_Y, PetscBool *changed_W, void *ctx)
{
  PetscFunctionBegin;
  PetscCallFortranVoidFunction((*(void (*)(SNESLineSearch *, Vec *, Vec *, Vec *, PetscBool *, PetscBool *, void *, PetscErrorCode *))(((PetscObject)linesearch)->fortran_func_pointers[2]))(&linesearch, &X, &Y, &W, changed_Y, changed_W, ctx, &ierr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN void sneslinesearchsetprecheck_(SNESLineSearch *linesearch, void (*func)(SNESLineSearch *, Vec *, Vec *, PetscBool *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*linesearch, 3);
  ((PetscObject)*linesearch)->fortran_func_pointers[1] = (PetscFortranCallbackFn *)func;

  *ierr = SNESLineSearchSetPreCheck(*linesearch, oursneslinesearchprecheck, ctx);
}

PETSC_EXTERN void sneslinesearchsetpostcheck_(SNESLineSearch *linesearch, void (*func)(SNESLineSearch *, Vec *, Vec *, Vec *, PetscBool *, PetscBool *, PetscErrorCode *, void *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*linesearch, 3);
  ((PetscObject)*linesearch)->fortran_func_pointers[2] = (PetscFortranCallbackFn *)func;

  *ierr = SNESLineSearchSetPostCheck(*linesearch, oursneslinesearchpostcheck, ctx);
}
