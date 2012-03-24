#include <petsc-private/fortranimpl.h>
#include <petsc-private/kspimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmkspsetcomputerhs_            DMKSPSETCOMPUTERHS_
#define dmkspsetcomputeoperators_      DMKSPSETCOMPUTEOPERATORS_
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmkspsetcomputerhs_            dmkspsetcomputerhs       /* zdmkspf.c */
#define dmkspsetcomputeoperators_      dmkspsetcomputeoperators /* zdmkspf */
#endif

static PetscErrorCode ourkspcomputerhs(KSP ksp,Vec b,void *ctx)
{
  PetscErrorCode ierr = 0;
  DM             dm;
  KSPDM          kdm;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr = DMKSPGetContext(dm,&kdm);CHKERRQ(ierr);
  (*(void (PETSC_STDCALL *)(KSP*,Vec*,void*,PetscErrorCode*))(kdm->fortran_func_pointers[0]))(&ksp,&b,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourkspcomputeoperators(KSP ksp,Mat A,Mat B,MatStructure *str,void *ctx)
{
  PetscErrorCode ierr = 0;
  DM             dm;
  KSPDM          kdm;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  ierr = DMKSPGetContext(dm,&kdm);CHKERRQ(ierr);
  (*(void (PETSC_STDCALL *)(KSP*,Mat*,Mat*,MatStructure*,void*,PetscErrorCode*))(kdm->fortran_func_pointers[1]))(&ksp,&A,&B,str,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

/* The counting for fortran_func_pointers is insanely brittle. We're putting these inside the base DM, but we have no
 * way to be sure there is room other than to grep the sources from src/dm (and any other possible client). Fortran
 * function pointers need an overhaul.
 */

PETSC_EXTERN_C void PETSC_STDCALL dmkspsetcomputerhs_(DM *dm,void (PETSC_STDCALL *func)(KSP*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  KSPDM kdm;
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = DMKSPGetContext(*dm,&kdm);
  if (!*ierr) {
    kdm->fortran_func_pointers[0] = (PetscVoidFunction)func;
    *ierr = DMKSPSetComputeRHS(*dm,ourkspcomputerhs,ctx);
  }
}

PETSC_EXTERN_C void PETSC_STDCALL dmkspsetcomputeoperators_(DM *dm,void (PETSC_STDCALL *func)(KSP*,Vec*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  KSPDM kdm;
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = DMKSPGetContext(*dm,&kdm);
  if (!*ierr) {
    kdm->fortran_func_pointers[1] = (PetscVoidFunction)func;
    *ierr = DMKSPSetComputeOperators(*dm,ourkspcomputeoperators,ctx);
  }
}

