#include <private/fortranimpl.h>
#include <petscdm.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmview_                      DMVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmview_                      dmview
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dmview_(DM *da,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = DMView(*da,v);
}
EXTERN_C_END
