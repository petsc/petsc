#include <petsc-private/fortranimpl.h>
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
  (*(void (PETSC_STDCALL *)(SNESLineSearch*,void*,PetscErrorCode*))(((PetscObject)linesearch)->fortran_func_pointers[0]))(&linesearch,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL sneslinesearchshellsetuserfunc_(SNESLineSearch *linesearch,
                                                    void (PETSC_STDCALL *func)(SNESLineSearch*,void*,PetscErrorCode*),
                                                    void *ctx,
                                                    PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*linesearch,3);
  ((PetscObject)*linesearch)->fortran_func_pointers[0] = (PetscVoidFunction)func;
  *ierr = SNESLineSearchShellSetUserFunc(*linesearch,oursneslinesearchshellfunction,ctx);
}

void PETSC_STDCALL sneslinesearchshellgetuserfunc_(SNESLineSearch *linesearch, void * func, void **ctx,PetscErrorCode *ierr)
{

  CHKFORTRANNULLINTEGER(ctx);
  *ierr = SNESLineSearchShellGetUserFunc(*linesearch,PETSC_NULL,ctx);
}
EXTERN_C_END
