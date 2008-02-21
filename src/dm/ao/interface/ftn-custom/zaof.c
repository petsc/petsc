
#include "private/fortranimpl.h"
#include "petscao.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define aoview_          AOVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define aoview_          aoview
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL aoview_(AO *ao,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = AOView(*ao,v);
}

EXTERN_C_END
