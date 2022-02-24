#include <petsc/private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define snesshellsetsolve_               SNESSHELLSETSOLVE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define snesshellsetsolve_               snesshellsetsolve
#endif

static PetscErrorCode oursnesshellsolve(SNES snes,Vec x)
{
  void (*func)(SNES*,Vec*,PetscErrorCode*);
  CHKERRQ(PetscObjectQueryFunction((PetscObject)snes,"SNESShellSolve_C",&func));
  PetscCheck(func,PetscObjectComm((PetscObject)snes),PETSC_ERR_USER,"SNESShellSetSolve() must be called before SNESSolve()");
  CHKERR_FORTRAN_VOID_FUNCTION(func(&snes,&x,&ierr));
  return 0;
}

PETSC_EXTERN void snesshellsetsolve_(SNES *snes,void (*func)(SNES*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectComposeFunction((PetscObject)*snes,"SNESShellSolve_C",(PetscVoidFunction)func);
  *ierr = SNESShellSetSolve(*snes,oursnesshellsolve);
}
