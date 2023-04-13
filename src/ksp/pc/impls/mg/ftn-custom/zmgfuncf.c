#include <petsc/private/fortranimpl.h>
#include <petscpc.h>
#include <petsc/private/pcmgimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcmgsetresidual_     PCMGSETRESIDUAL
  #define pcmgresidualdefault_ PCMGRESIDUALDEFAULT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcmgsetresidual_     pcmgsetresidual
  #define pcmgresidualdefault_ pcmgresidualdefault
#endif

typedef PetscErrorCode (*MVVVV)(Mat, Vec, Vec, Vec);
static PetscErrorCode ourresidualfunction(Mat mat, Vec b, Vec x, Vec R)
{
  PetscCallFortranVoidFunction((*(void (*)(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[0]))(&mat, &b, &x, &R, &ierr));
  return PETSC_SUCCESS;
}

PETSC_EXTERN void pcmgresidualdefault_(Mat *mat, Vec *b, Vec *x, Vec *r, PetscErrorCode *ierr)
{
  *ierr = PCMGResidualDefault(*mat, *b, *x, *r);
}

PETSC_EXTERN void pcmgsetresidual_(PC *pc, PetscInt *l, PetscErrorCode (*residual)(Mat *, Vec *, Vec *, Vec *, PetscErrorCode *), Mat *mat, PetscErrorCode *ierr)
{
  MVVVV rr;
  if ((PetscVoidFunction)residual == (PetscVoidFunction)pcmgresidualdefault_) rr = PCMGResidualDefault;
  else {
    PetscObjectAllocateFortranPointers(*mat, 1);
    /*  Attach the residual computer to the Mat, this is not ideal but the only object/context passed in the residual computer */
    ((PetscObject)*mat)->fortran_func_pointers[0] = (PetscVoidFunction)residual;

    rr = ourresidualfunction;
  }
  *ierr = PCMGSetResidual(*pc, *l, rr, *mat);
}
