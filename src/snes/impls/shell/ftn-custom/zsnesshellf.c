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
  void (PETSC_STDCALL *func)(SNES*,Vec*,PetscErrorCode*);
  ierr = PetscObjectQueryFunction((PetscObject)snes,"SNESShellSolve_C",(PetscVoidFunction*)&func);CHKERRQ(ierr);
  if (!func) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_USER,"SNESShellSetSolve() must be called before SNESSolve()");
  func(&snes,&x,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN
void PETSC_STDCALL snesshellsetsolve_(SNES *snes,void (PETSC_STDCALL *func)(SNES*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectComposeFunctionDynamic((PetscObject)*snes,"SNESShellSolve_C",PETSC_NULL,(PetscVoidFunction)func);
  *ierr = SNESShellSetSolve(*snes,oursnesshellsolve);
}
EXTERN_C_END
