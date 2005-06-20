#include "zpetsc.h"
#include "petscsys.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawzoom_            PETSCDRAWZOOM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawzoom_            petscdrawzoom
#endif

typedef void (PETSC_STDCALL *FCN)(PetscDraw*,void*,PetscErrorCode*); /* force argument to next function to not be extern C*/
static FCN f1;

static PetscErrorCode ourdrawzoom(PetscDraw draw,void *ctx)
{
  PetscErrorCode ierr = 0;

  (*f1)(&draw,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL petscdrawzoom_(PetscDraw *draw,FCN f,void *ctx,PetscErrorCode *ierr)
{
  f1      = f;
  *ierr = PetscDrawZoom(*draw,ourdrawzoom,ctx);
}

EXTERN_C_END
