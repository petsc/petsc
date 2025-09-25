#include <petsc/private/ftnimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define sneslinesearchshellsetapply_ SNESLINESEARCHSHELLSETAPPLY
  #define sneslinesearchshellgetapply_ SNESLINESEARCHSHELLGETAPPLY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define sneslinesearchshellsetapply_ sneslinesearchshellsetapply
  #define sneslinesearchshellgetapply_ sneslinesearchshellgetapply
#endif

static PetscErrorCode oursneslinesearchshellfunction(SNESLineSearch linesearch, void *ctx)
{
  PetscFunctionBegin;
  PetscCallFortranVoidFunction((*(void (*)(SNESLineSearch *, void *, PetscErrorCode *))(((PetscObject)linesearch)->fortran_func_pointers[0]))(&linesearch, ctx, &ierr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN void sneslinesearchshellsetapply_(SNESLineSearch *linesearch, void (*func)(SNESLineSearch *, void *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*linesearch, 3);
  ((PetscObject)*linesearch)->fortran_func_pointers[0] = (PetscFortranCallbackFn *)func;

  *ierr = SNESLineSearchShellSetApply(*linesearch, oursneslinesearchshellfunction, ctx);
}

PETSC_EXTERN void sneslinesearchshellgetapply_(SNESLineSearch *linesearch, void *func, void **ctx, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(ctx);
  *ierr = SNESLineSearchShellGetApply(*linesearch, NULL, ctx);
}
