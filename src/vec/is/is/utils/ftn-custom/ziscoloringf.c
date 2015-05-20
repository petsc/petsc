#include <petsc/private/fortranimpl.h>
#include <petscis.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define iscoloringview_        ISCOLORINGVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define iscoloringview_        iscoloringview
#endif

PETSC_EXTERN void PETSC_STDCALL iscoloringview_(ISColoring *iscoloring,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = ISColoringView(*iscoloring,v);
}
