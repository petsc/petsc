#include <petsc/private/fortranimpl.h>
#include <petsc/private/kspimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmkspsetcomputerhs_            DMKSPSETCOMPUTERHS
#define dmkspsetcomputeinitialguess_   DMKSPSETCOMPUTEINITIALGUESS
#define dmkspsetcomputeoperators_      DMKSPSETCOMPUTEOPERATORS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmkspsetcomputerhs_            dmkspsetcomputerhs          /* zdmkspf.c */
#define dmkspsetcomputeinitialguess_   dmkspsetcomputeinitialguess /* zdmkspf.c */
#define dmkspsetcomputeoperators_      dmkspsetcomputeoperators    /* zdmkspf */
#endif

static PetscErrorCode ourkspcomputerhs(KSP ksp,Vec b,void *ctx)
{
  PetscErrorCode ierr = 0;
  DM             dm;
  DMKSP          kdm;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr = DMGetDMKSP(dm,&kdm);CHKERRQ(ierr);
  (*(void (*)(KSP*,Vec*,void*,PetscErrorCode*))(kdm->fortran_func_pointers[0]))(&ksp,&b,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourkspcomputeinitialguess(KSP ksp,Vec b,void *ctx)
{
  PetscErrorCode ierr = 0;
  DM             dm;
  DMKSP          kdm;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr = DMGetDMKSP(dm,&kdm);CHKERRQ(ierr);
  (*(void (*)(KSP*,Vec*,void*,PetscErrorCode*))(kdm->fortran_func_pointers[2]))(&ksp,&b,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourkspcomputeoperators(KSP ksp,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr = 0;
  DM             dm;
  DMKSP          kdm;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr = DMGetDMKSP(dm,&kdm);CHKERRQ(ierr);
  (*(void (*)(KSP*,Mat*,Mat*,void*,PetscErrorCode*))(kdm->fortran_func_pointers[1]))(&ksp,&A,&B,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

/* The counting for fortran_func_pointers is insanely brittle. We're putting these inside the base DM, but we have no
 * way to be sure there is room other than to grep the sources from src/dm (and any other possible client). Fortran
 * function pointers need an overhaul.
 */

PETSC_EXTERN void dmkspsetcomputerhs_(DM *dm,void (*func)(KSP*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  DMKSP kdm;
  *ierr = DMGetDMKSP(*dm,&kdm);
  if (!*ierr) {
    kdm->fortran_func_pointers[0] = (PetscVoidFunction)func;
    *ierr = DMKSPSetComputeRHS(*dm,ourkspcomputerhs,ctx);
  }
}

PETSC_EXTERN void dmkspsetcomputeinitialguess_(DM *dm,void (*func)(KSP*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  DMKSP kdm;
  *ierr = DMGetDMKSP(*dm,&kdm);
  if (!*ierr) {
    kdm->fortran_func_pointers[2] = (PetscVoidFunction)func;

    *ierr = DMKSPSetComputeInitialGuess(*dm,ourkspcomputeinitialguess,ctx);
  }
}

PETSC_EXTERN void dmkspsetcomputeoperators_(DM *dm,void (*func)(KSP*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  DMKSP kdm;
  *ierr = DMGetDMKSP(*dm,&kdm);
  if (!*ierr) {
    kdm->fortran_func_pointers[1] = (PetscVoidFunction)func;
    *ierr = DMKSPSetComputeOperators(*dm,ourkspcomputeoperators,ctx);
  }
}

