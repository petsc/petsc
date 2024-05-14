#include <petsc/private/fortranimpl.h>
#include <petscfe.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscfeview_ PETSCFEVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscfeview_ petscfeview
#endif

PETSC_EXTERN void petscfeview_(PetscFE *fe, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscFEView(*fe, v);
}
