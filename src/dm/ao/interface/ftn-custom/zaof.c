
#include <petsc-private/fortranimpl.h>
#include <petscao.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define aoview_          AOVIEW
#define aodestroy_       AODESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define aoview_          aoview
#define aodestroy_       aodestroy
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL aoview_(AO *ao,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = AOView(*ao,v);
}

void PETSC_STDCALL aodestroy_(AO *ao, int *__ierr )
{
  *__ierr = AODestroy(ao);
}

EXTERN_C_END
