#include "private/fortranimpl.h"
#include "petscpc.h"
#include "petscmg.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcmgsetresidual_           PCMGSETRESIDUAL
#define pcmgdefaultresidual_       PCMGDEFAULTRESIDUAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcmgsetresidual_           pcmgsetresidual
#define pcmgdefaultresidual_       pcmgdefaultresidual
#endif

typedef PetscErrorCode (*MVVVV)(Mat,Vec,Vec,Vec);
static PetscErrorCode ourresidualfunction(Mat mat,Vec b,Vec x,Vec R)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(Mat*,Vec*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[0]))(&mat,&b,&x,&R,&ierr);
  return 0;
}

EXTERN_C_BEGIN
void pcmgdefaultresidual_(Mat *mat,Vec *b,Vec *x,Vec *r, PetscErrorCode *ierr)
{
  *ierr = PCMGDefaultResidual(*mat,*b,*x,*r);
}

void PETSC_STDCALL pcmgsetresidual_(PC *pc,PetscInt *l,PetscErrorCode (*residual)(Mat*,Vec*,Vec*,Vec*,PetscErrorCode*),Mat *mat, PetscErrorCode *ierr)
{
  MVVVV rr;
  if ((PetscVoidFunction)residual == (PetscVoidFunction)pcmgdefaultresidual_) rr = PCMGDefaultResidual;
  else {
    PetscObjectAllocateFortranPointers(*mat,1);
    /*  Attach the residual computer to the Mat, this is not ideal but the only object/context passed in the residual computer */
    ((PetscObject)*mat)->fortran_func_pointers[0] = (PetscVoidFunction)residual;
    rr = ourresidualfunction;
  }
  *ierr = PCMGSetResidual(*pc,*l,rr,*mat);
}

EXTERN_C_END
