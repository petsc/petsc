#include <petsc/private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define snesshellsetsolve_               SNESSHELLSETSOLVE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define snesshellsetsolve_               snesshellsetsolve
#endif

static PetscErrorCode oursnesshellsolve(SNES snes,Vec x)
{
  PetscErrorCode ierr = 0;
  void (*func)(SNES*,Vec*,PetscErrorCode*);
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESShellSolve_C",&func);CHKERRQ(ierr);
  PetscAssertFalse(!func,PetscObjectComm((PetscObject)snes),PETSC_ERR_USER,"SNESShellSetSolve() must be called before SNESSolve()");
  func(&snes,&x,&ierr);CHKERRQ(ierr);
  return 0;
}

PETSC_EXTERN void snesshellsetsolve_(SNES *snes,void (*func)(SNES*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectComposeFunction((PetscObject)*snes,"SNESShellSolve_C",(PetscVoidFunction)func);
  *ierr = SNESShellSetSolve(*snes,oursnesshellsolve);
}
