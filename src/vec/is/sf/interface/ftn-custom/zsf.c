#include <petsc/private/fortranimpl.h>
#include <petsc/private/sfimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define sfview_ PETSCSFVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sfview_ petscsfview
#endif

PETSC_EXTERN void PETSC_STDCALL petscsfview_(PetscSF *sf, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscSFView(*sf, v);
}
