#include <petsc/private/ftnimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matmffdsetfunction_ MATMFFDSETFUNCTION
  #define matmffdsetbase_     MATMFFDSETBASE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matmffdsetfunction_ matmffdsetfunction
  #define matmffdsetbase_     matmffdsetbase
#endif

static PetscErrorCode ourmatmffdfunction(void *ctx, Vec x, Vec f)
{
  Mat mat = (Mat)ctx;
  PetscCallFortranVoidFunction((*(void (*)(void *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)mat)->fortran_func_pointers[0]))((void *)(PETSC_UINTPTR_T)((PetscObject)mat)->fortran_func_pointers[1], &x, &f, &ierr));
  return PETSC_SUCCESS;
}

PETSC_EXTERN void matmffdsetfunction_(Mat *mat, void (*func)(void *, Vec *, Vec *, PetscErrorCode *), void *ctx, PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*mat, 2);
  ((PetscObject)*mat)->fortran_func_pointers[0] = (PetscFortranCallbackFn *)func;
  ((PetscObject)*mat)->fortran_func_pointers[1] = (PetscFortranCallbackFn *)(PETSC_UINTPTR_T)ctx;

  *ierr = MatMFFDSetFunction(*mat, ourmatmffdfunction, *mat);
}
