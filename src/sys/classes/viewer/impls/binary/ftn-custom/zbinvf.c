#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerfilesetmode_         PETSCVIEWERFILESETMODE
  #define petscviewerbinarygetdescriptor_ PETSCVIEWERBINARYGETDESCRIPTOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerfilesetmode_         petscviewerfilesetmode
  #define petscviewerbinarygetdescriptor_ petscviewerbinarygetdescriptor
#endif

PETSC_EXTERN void petscviewerfilesetmode_(PetscViewer *viewer, PetscFileMode *type, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscViewerFileSetMode(v, *type);
}

PETSC_EXTERN void petscviewerbinarygetdescriptor_(PetscViewer *viewer, int *fd, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscViewerBinaryGetDescriptor(v, fd);
}
