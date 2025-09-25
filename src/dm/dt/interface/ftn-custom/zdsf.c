#include <petsc/private/ftnimpl.h>
#include <petscds.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscdssetriemannsolver_ PETSCDSSETRIEMANNSOLVER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscdssetriemannsolver_ petscdssetriemannsolver
#endif

static PetscFortranCallbackId riemannsolver;

// We can't use PetscObjectUseFortranCallback() because this function returns void
static void ourriemannsolver(PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], PetscInt numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx)
{
  void (*func)(PetscInt *dim, PetscInt *Nf, const PetscReal x[], const PetscReal n[], const PetscScalar uL[], const PetscScalar uR[], const PetscInt *numConstants, const PetscScalar constants[], PetscScalar flux[], void *ctx);
  void *_ctx;
  PetscCallAbort(PETSC_COMM_SELF, PetscObjectGetFortranCallback((PetscObject)ctx, PETSC_FORTRAN_CALLBACK_CLASS, riemannsolver, (PetscFortranCallbackFn **)&func, &_ctx));
  if (func) (*func)(&dim, &Nf, x, n, uL, uR, &numConstants, constants, flux, _ctx);
}

PETSC_EXTERN void petscdssetriemannsolver_(PetscDS *prob, PetscInt *f, void (*rs)(PetscInt *, PetscInt *, PetscReal *, PetscReal *, PetscScalar *, PetscScalar *, PetscInt *, PetscScalar *, PetscScalar *, void *, PetscErrorCode *), PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*prob, PETSC_FORTRAN_CALLBACK_CLASS, &riemannsolver, (PetscFortranCallbackFn *)rs, NULL);
  if (*ierr) return;
  *ierr = PetscDSSetRiemannSolver(*prob, *f, ourriemannsolver);
  if (*ierr) return;
}
