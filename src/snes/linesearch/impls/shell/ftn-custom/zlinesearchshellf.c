#include <private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsclinesearchshellsetuserfunc_          PETSCLINESEARCHSHELLSETUSERFUNC
#define petsclinesearchshellgetuserfunc_          PETSCLINESEARCHSHELLGETUSERFUNC
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsclinesearchshellsetuserfunc_          petsclinesearchshellsetuserfunc
#define petsclinesearchshellgetuserfunc_          petsclinesearchshellgetuserfunc
#endif

static PetscErrorCode ourpetsclinesearchshellfunction(PetscLineSearch linesearch, void *ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(PetscLineSearch*,void*,PetscErrorCode*))(((PetscObject)linesearch)->fortran_func_pointers[0]))(&linesearch,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL petsclinesearchshellsetuserfunc_(PetscLineSearch *linesearch,
                                                    void (PETSC_STDCALL *func)(PetscLineSearch*,void*,PetscErrorCode*),
                                                    void *ctx,
                                                    PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*linesearch,3);
  ((PetscObject)*linesearch)->fortran_func_pointers[0] = (PetscVoidFunction)func;
  *ierr = PetscLineSearchShellSetUserFunc(*linesearch,ourpetsclinesearchshellfunction,ctx);
}

void PETSC_STDCALL petsclinesearchshellgetuserfunc_(PetscLineSearch *linesearch, void * func, void **ctx,PetscErrorCode *ierr)
{

  CHKFORTRANNULLINTEGER(ctx);
  *ierr = PetscLineSearchShellGetUserFunc(*linesearch,PETSC_NULL,ctx);
}
EXTERN_C_END
