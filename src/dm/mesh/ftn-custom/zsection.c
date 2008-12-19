#include "private/fortranimpl.h"
#include "petscmesh.h"
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define sectionrealview_ SECTIONREALVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sectionrealview_ sectionrealview
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL sectionrealview_(SectionReal *x,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = SectionRealView(*x,v);
}
EXTERN_C_END
