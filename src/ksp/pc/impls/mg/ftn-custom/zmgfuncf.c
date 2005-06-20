#include "zpetsc.h"
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
extern void PETSC_STDCALL pcmgdefaultresidual_(Mat*,Vec*,Vec*,Vec*,PetscErrorCode*);

void PETSC_STDCALL pcmgsetresidual_(PC *pc,PetscInt *l,PetscErrorCode (*residual)(Mat*,Vec*,Vec*,Vec*,PetscErrorCode*),Mat *mat, PetscErrorCode *ierr)
{
  MVVVV rr;
  if ((FCNVOID)residual == (FCNVOID)pcmgdefaultresidual_) rr = PCMGDefaultResidual;
  else {
    if (!((PetscObject)*mat)->fortran_func_pointers) {
      *ierr = PetscMalloc(1*sizeof(void*),&((PetscObject)*mat)->fortran_func_pointers);
    }
    ((PetscObject)*mat)->fortran_func_pointers[0] = (FCNVOID)residual;
    rr = ourresidualfunction;
  }
  *ierr = PCMGSetResidual(*pc,*l,rr,*mat);
}

EXTERN_C_END
