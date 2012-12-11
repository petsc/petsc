#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawzoom_            PETSCDRAWZOOM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawzoom_            petscdrawzoom
#endif

typedef void (PETSC_STDCALL *FCN)(PetscDraw*,void*,PetscErrorCode*); /* force argument to next function to not be extern C*/

static PetscErrorCode ourdrawzoom(PetscDraw draw,void *ctx)
{
  PetscErrorCode ierr = 0;

  (*(void (PETSC_STDCALL *)(PetscDraw*,void*,PetscErrorCode*))(((PetscObject)draw)->fortran_func_pointers[0]))(&draw,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL petscdrawzoom_(PetscDraw *draw,FCN f,void *ctx,PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*draw,1);
  ((PetscObject)*draw)->fortran_func_pointers[0] = (PetscVoidFunction)f;
  *ierr = PetscDrawZoom(*draw,ourdrawzoom,ctx);
}

EXTERN_C_END
