#include <petsc/private/ftnimpl.h>
#include <petscpc.h>
#include <petsc/private/pcmgimpl.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define pcmgsetresidual_              PCMGSETRESIDUAL
  #define pcmgresidualdefault_          PCMGRESIDUALDEFAULT
  #define pcmgsetresidualtranspose_     PCMGSETRESIDUALTRANSPOSE
  #define pcmgresidualtransposedefault_ PCMGRESIDUALTRANSPOSEDEFAULT
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define pcmgsetresidual_              pcmgsetresidual
  #define pcmgresidualdefault_          pcmgresidualdefault
  #define pcmgsetresidualtranspose_     pcmgsetresidualtranspose
  #define pcmgresidualtransposedefault_ pcmgresidualtransposedefault
#endif

typedef PetscErrorCode (*MVVVV)(Mat, Vec, Vec, Vec);

static struct {
  PetscFortranCallbackId residual;
  PetscFortranCallbackId residualtranspose;
} _cb;

static PetscErrorCode ourresidualfunction(Mat mat, Vec b, Vec x, Vec R)
{
  PetscObjectUseFortranCallback(mat, _cb.residual, (Mat *, Vec *, Vec *, Vec *, PetscErrorCode *), (&mat, &b, &x, &R, &ierr));
}

PETSC_EXTERN void pcmgresidualdefault_(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *);

PETSC_EXTERN void pcmgsetresidual_(PC *pc, PetscInt *l, void (*residual)(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *), Mat *mat, PetscErrorCode *ierr)
{
  MVVVV rr;

  CHKFORTRANNULLFUNCTION(residual);
  if (!residual) rr = NULL;
  else if (residual == pcmgresidualdefault_) rr = PCMGResidualDefault;
  else {
    /* The Mat is the only object passed to the residual computer. A keyed callback is used so that this does not collide with
       the Fortran callbacks of a MATSHELL or MATMFFD, which are stored in fortran_func_pointers[] */
    *ierr = PetscObjectSetFortranCallback((PetscObject)*mat, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.residual, (PetscFortranCallbackFn *)residual, NULL);
    if (*ierr) return;
    rr = ourresidualfunction;
  }
  *ierr = PCMGSetResidual(*pc, *l, rr, *mat);
}

static PetscErrorCode ourresidualtransposefunction(Mat mat, Vec b, Vec x, Vec R)
{
  PetscObjectUseFortranCallback(mat, _cb.residualtranspose, (Mat *, Vec *, Vec *, Vec *, PetscErrorCode *), (&mat, &b, &x, &R, &ierr));
}

PETSC_EXTERN void pcmgresidualtransposedefault_(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *);

PETSC_EXTERN void pcmgsetresidualtranspose_(PC *pc, PetscInt *l, void (*residualt)(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *), Mat *mat, PetscErrorCode *ierr)
{
  MVVVV rr;

  CHKFORTRANNULLFUNCTION(residualt);
  if (!residualt) rr = NULL;
  else if (residualt == pcmgresidualtransposedefault_) rr = PCMGResidualTransposeDefault;
  else {
    *ierr = PetscObjectSetFortranCallback((PetscObject)*mat, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.residualtranspose, (PetscFortranCallbackFn *)residualt, NULL);
    if (*ierr) return;
    rr = ourresidualtransposefunction;
  }
  *ierr = PCMGSetResidualTranspose(*pc, *l, rr, *mat);
}
