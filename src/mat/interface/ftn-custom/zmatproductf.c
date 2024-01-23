#include <petsc/private/fortranimpl.h>
#include <petscmat.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matproductview_         MATPRODUCTVIEW
  #define matproductsetalgorithm_ MATPRODUCTSETALGORITHM
  #define matproductgetalgorithm_ MATPRODUCTGETALGORITHM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matproductview_         matproductview
  #define matproductsetalgorithm_ matproductsetalgorithm
  #define matproductgetalgorithm_ matproductgetalgorithm
#endif

PETSC_EXTERN void matproductview_(Mat *mat, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = MatProductView(*mat, v);
}

PETSC_EXTERN void matproductsetalgorithm_(Mat *mat, char *algorithm, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(algorithm, len, t);
  *ierr = MatProductSetAlgorithm(*mat, t);
  if (*ierr) return;
  FREECHAR(algorithm, t);
}

PETSC_EXTERN void matproductgetalgorithm_(Mat *mat, char *algorithm, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *talgo;

  *ierr = MatProductGetAlgorithm(*mat, &talgo);
  if (*ierr) return;
  if (algorithm != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(algorithm, talgo, len);
    if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE, algorithm, len);
}
