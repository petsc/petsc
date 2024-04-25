#include <petsc/private/fortranimpl.h>
#include <petscksp.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define kspguessview_ KSPGUESSVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define kspguessview_ kspguessview
#endif

PETSC_EXTERN void kspguessview_(KSPGuess *kspguess, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = KSPGuessView(*kspguess, v);
}
