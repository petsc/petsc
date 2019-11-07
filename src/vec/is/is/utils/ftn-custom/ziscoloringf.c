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

PETSC_EXTERN void PETSC_STDCALL iscoloringview_(ISColoring *iscoloring,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = ISColoringView(*iscoloring,v);
}

PETSC_EXTERN void PETSC_STDCALL iscoloringviewfromoptions_(ISColoring *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = ISColoringViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
