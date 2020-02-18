
#include <petsc/private/fortranimpl.h>
#include <petscao.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define aoview_          AOVIEW
#define aosettype_       AOSETTYPE
#define aoviewfromoptions_ AOVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define aoview_          aoview
#define aosettype_       aosettype
#define aoviewfromoptions_ aoviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL aoview_(AO *ao,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = AOView(*ao,v);
}

PETSC_EXTERN void PETSC_STDCALL aosettype_(AO *ao,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = AOSetType(*ao,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL aoviewfromoptions_(AO *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = AOViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}
