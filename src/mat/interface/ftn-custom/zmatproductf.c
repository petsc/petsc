#include <petsc/private/fortranimpl.h>
#include <petscmat.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matproductview_ MATPRODUCTVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matproductview_ matproductview
#endif

PETSC_EXTERN void matproductview_(Mat *mat, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = MatProductView(*mat, v);
}
