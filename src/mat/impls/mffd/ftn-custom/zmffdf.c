#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matmffdsetfunction_      MATMFFDSETFUNCTION
  #define matmffdsettype_          MATMFFDSETTYPE
  #define matmffdsetoptionsprefix_ MATMFFDSETOPTIONSPREFIX
  #define matmffdsetbase_          MATMFFDSETBASE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matmffdsetfunction_      matmffdsetfunction
  #define matmffdsettype_          matmffdsettype
  #define matmffdsetoptionsprefix_ matmffdsetoptionsprefix
  #define matmffdsetbase_          matmffdsetbase
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
  ((PetscObject)*mat)->fortran_func_pointers[0] = (PetscVoidFunction)func;
  ((PetscObject)*mat)->fortran_func_pointers[1] = (PetscVoidFunction)(PETSC_UINTPTR_T)ctx;

  *ierr = MatMFFDSetFunction(*mat, ourmatmffdfunction, *mat);
}

PETSC_EXTERN void matmffdsettype_(Mat *mat, char *ftype, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(ftype, len, t);
  *ierr = MatMFFDSetType(*mat, t);
  if (*ierr) return;
  FREECHAR(ftype, t);
}

PETSC_EXTERN void matmffdsetoptionsprefix_(Mat *mat, char *prefix, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(prefix, len, t);
  *ierr = MatMFFDSetOptionsPrefix(*mat, t);
  if (*ierr) return;
  FREECHAR(prefix, t);
}
