#include <petsc/private/fortranimpl.h>
#include <petsc/private/snesimpl.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmsnessetjacobian_      DMSNESSETJACOBIAN
#define dmsnessetfunction_      DMSNESSETFUNCTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmsnessetjacobian_      dmsnessetjacobian
#define dmsnessetfunction_      dmsnessetfunction
#endif

static struct {
  PetscFortranCallbackId snesfunction;
  PetscFortranCallbackId snesjacobian;
} _cb;

static PetscErrorCode ourj(SNES snes, Vec X, Mat J, Mat P, void *ptr)
{
  void (*func)(SNES*,Vec*,Mat*,Mat*,void*,PetscErrorCode*),*ctx;
  DM dm;
  DMSNES sdm;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetDMSNES(dm, &sdm));
  CHKERRQ(PetscObjectGetFortranCallback((PetscObject) sdm, PETSC_FORTRAN_CALLBACK_SUBTYPE, _cb.snesjacobian, (PetscVoidFunction *) &func, &ctx));
  CHKERR_FORTRAN_VOID_FUNCTION((*func)(&snes, &X, &J, &P, ctx, &ierr));
  PetscFunctionReturn(0);
}

PETSC_EXTERN void dmsnessetjacobian_(DM *dm, void (*jac)(DM*,Vec*,Mat*,Mat*,void*,PetscErrorCode*), void *ctx, PetscErrorCode *ierr)
{
  DMSNES sdm;

  *ierr = DMGetDMSNESWrite(*dm, &sdm); if (*ierr) return;
  *ierr = PetscObjectSetFortranCallback((PetscObject) sdm, PETSC_FORTRAN_CALLBACK_SUBTYPE, &_cb.snesjacobian, (PetscVoidFunction) jac, ctx); if (*ierr) return;
  *ierr = DMSNESSetJacobian(*dm, ourj, NULL);
}

static PetscErrorCode ourf(SNES snes, Vec X, Vec F, void *ptr)
{
  void (*func)(SNES*,Vec*,Vec*,void*,PetscErrorCode*), *ctx;
  DM dm;
  DMSNES sdm;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes,&dm));
  CHKERRQ(DMGetDMSNES(dm, &sdm));
  CHKERRQ(PetscObjectGetFortranCallback((PetscObject) sdm, PETSC_FORTRAN_CALLBACK_SUBTYPE, _cb.snesfunction, (PetscVoidFunction *) &func, &ctx));
  CHKERR_FORTRAN_VOID_FUNCTION((*func)(&snes, &X, &F, ctx, &ierr));
  PetscFunctionReturn(0);
}

PETSC_EXTERN void dmsnessetfunction_(DM *dm, void (*func)(SNES*,Vec*,Vec*,void*,PetscErrorCode*), void *ctx, PetscErrorCode *ierr)
{
  DMSNES sdm;

  *ierr = DMGetDMSNESWrite(*dm, &sdm); if (*ierr) return;
  *ierr = PetscObjectSetFortranCallback((PetscObject) sdm, PETSC_FORTRAN_CALLBACK_SUBTYPE, &_cb.snesfunction, (PetscVoidFunction) func, ctx); if (*ierr) return;
  *ierr = DMSNESSetFunction(*dm, ourf, NULL);
}
