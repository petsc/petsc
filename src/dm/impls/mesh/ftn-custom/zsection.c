#include <private/fortranimpl.h>
#include <petscdmmesh.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define sectionrealview_       SECTIONREALVIEW
#define sectionintview_        SECTIONINTVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sectionrealview_       sectionrealview
#define sectionintview_        sectionintview
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL sectionrealview_(SectionReal *x,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = SectionRealView(*x,v);
}
void PETSC_STDCALL sectionintview_(SectionInt *x,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = SectionIntView(*x,v);
}
EXTERN_C_END
