#include <petsc/private/fortranimpl.h>
#include <petscis.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define iscoloringview_        ISCOLORINGVIEW
#define iscoloringviewfromoptions_ ISCOLORINGVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define iscoloringview_        iscoloringview
#define iscoloringviewfromoptions_ iscoloringviewfromoptions
#endif

PETSC_EXTERN void iscoloringview_(ISColoring *iscoloring,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = ISColoringView(*iscoloring,v);
}

PETSC_EXTERN void iscoloringviewfromoptions_(ISColoring *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = ISColoringViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
