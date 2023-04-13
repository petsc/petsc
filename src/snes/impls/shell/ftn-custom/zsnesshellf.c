#include <petsc/private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define snesshellsetsolve_ SNESSHELLSETSOLVE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define snesshellsetsolve_ snesshellsetsolve
#endif

static PetscErrorCode oursnesshellsolve(SNES snes, Vec x)
{
  void (*func)(SNES *, Vec *, PetscErrorCode *);
  PetscCall(PetscObjectQueryFunction((PetscObject)snes, "SNESShellSolve_C", &func));
  PetscCheck(func, PetscObjectComm((PetscObject)snes), PETSC_ERR_USER, "SNESShellSetSolve() must be called before SNESSolve()");
  PetscCallFortranVoidFunction(func(&snes, &x, &ierr));
  return PETSC_SUCCESS;
}

PETSC_EXTERN void snesshellsetsolve_(SNES *snes, void (*func)(SNES *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  *ierr = PetscObjectComposeFunction((PetscObject)*snes, "SNESShellSolve_C", (PetscVoidFunction)func);
  if (*ierr) return;
  *ierr = SNESShellSetSolve(*snes, oursnesshellsolve);
}
