#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define daview_                      DMDAVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define daview_                      daview
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL daview_(DM *da,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = DMView(*da,v);
}
EXTERN_C_END
