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
static PetscErrorCode ourresidualfunction(Mat mat, Vec b, Vec x, Vec R)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[0]))(&mat, &b, &x, &R, &ierr));
  return PETSC_SUCCESS;
}

PETSC_EXTERN void pcmgresidualdefault_(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *);

PETSC_EXTERN void pcmgsetresidual_(PC *pc, PetscInt *l, void (*residual)(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *), Mat *mat, PetscErrorCode *ierr)
{
  MVVVV rr;
  if (residual == pcmgresidualdefault_) rr = PCMGResidualDefault;
  else {
    PetscObjectAllocateFortranPointers(*mat, 1);
    /*  Attach the residual computer to the Mat, this is not ideal but the only object/context passed in the residual computer */
    ((PetscObject)*mat)->fortran_func_pointers[0] = (PetscFortranCallbackFn *)residual;

    rr = ourresidualfunction;
  }
  *ierr = PCMGSetResidual(*pc, *l, rr, *mat);
}

static struct {
  PetscFortranCallbackId residualtranspose;
} _cb;

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
    /* The Mat is the only object passed to the residual computer. A keyed callback is used so that this does not collide with
       the Fortran callbacks of a MATSHELL or MATMFFD, which are stored in fortran_func_pointers[] */
    *ierr = PetscObjectSetFortranCallback((PetscObject)*mat, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.residualtranspose, (PetscFortranCallbackFn *)residualt, NULL);
    if (*ierr) return;
    rr = ourresidualtransposefunction;
  }
  *ierr = PCMGSetResidualTranspose(*pc, *l, rr, *mat);
}
