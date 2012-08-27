#include <petsc-private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define snesshellsetsolve_               SNESSHELLSETSOLVE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define snesshellsetsolve_               snesshellsetsolve
#endif

static PetscErrorCode oursnesshellsolve(SNES snes,Vec x)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(SNES*,Vec*,PetscErrorCode*))(((PetscObject)snes)->fortran_func_pointers[12]))(&snes,&x,&ierr);CHKERRQ(ierr);
  return 0;
}
void PETSC_STDCALL snesshellsetsolve_(SNES *snes,void (PETSC_STDCALL *func)(SNES*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*snes,14);
  ((PetscObject)*snes)->fortran_func_pointers[12] = (PetscVoidFunction)func;
  *ierr = SNESShellSetSolve(*snes,oursnesshellsolve);
}


