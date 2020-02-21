#include <petsc/private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define sneslinesearchshellsetuserfunc_          SNESLINESEARCHSHELLSETUSERFUNC
#define sneslinesearchshellgetuserfunc_          SNESLINESEARCHSHELLGETUSERFUNC
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sneslinesearchshellsetuserfunc_          sneslinesearchshellsetuserfunc
#define sneslinesearchshellgetuserfunc_          sneslinesearchshellgetuserfunc
#endif

static PetscErrorCode oursneslinesearchshellfunction(SNESLineSearch linesearch, void *ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (*)(SNESLineSearch*,void*,PetscErrorCode*))(((PetscObject)linesearch)->fortran_func_pointers[0]))(&linesearch,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

PETSC_EXTERN void sneslinesearchshellsetuserfunc_(SNESLineSearch *linesearch,void (*func)(SNESLineSearch*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*linesearch,3);
  ((PetscObject)*linesearch)->fortran_func_pointers[0] = (PetscVoidFunction)func;

  *ierr = SNESLineSearchShellSetUserFunc(*linesearch,oursneslinesearchshellfunction,ctx);
}

PETSC_EXTERN void sneslinesearchshellgetuserfunc_(SNESLineSearch *linesearch, void * func, void **ctx,PetscErrorCode *ierr)
{

  CHKFORTRANNULLINTEGER(ctx);
  *ierr = SNESLineSearchShellGetUserFunc(*linesearch,NULL,ctx);
}
