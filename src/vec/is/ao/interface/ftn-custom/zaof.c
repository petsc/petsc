
#include <petsc/private/fortranimpl.h>
#include <petscao.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define aoview_          AOVIEW
#define aosettype_       AOSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define aoview_          aoview
#define aosettype_       aosettype
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
  *ierr = AOSetType(*ao,t);
  FREECHAR(type,t);
}
