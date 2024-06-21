#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerbinarygetdescriptor_ PETSCVIEWERBINARYGETDESCRIPTOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerbinarygetdescriptor_ petscviewerbinarygetdescriptor
#endif

PETSC_EXTERN void petscviewerbinarygetdescriptor_(PetscViewer *viewer, int *fd, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscViewerBinaryGetDescriptor(v, fd);
}
