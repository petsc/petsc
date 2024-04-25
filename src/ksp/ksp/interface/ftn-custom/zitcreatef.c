#include <petsc/private/fortranimpl.h>
#include <petscksp.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define kspview_            KSPVIEW
  #define kspviewfromoptions_ KSPVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define kspview_            kspview
  #define kspviewfromoptions_ kspviewfromoptions
#endif

PETSC_EXTERN void kspview_(KSP *ksp, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = KSPView(*ksp, v);
}

PETSC_EXTERN void kspviewfromoptions_(KSP *ao, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = KSPViewFromOptions(*ao, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}
