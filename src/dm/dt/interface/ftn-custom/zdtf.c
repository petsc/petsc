#include <petsc/private/fortranimpl.h>
#include <petscdt.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscquadratureview_ PETSCQUADRATUREVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscquadratureview_ petscquadratureview
#endif

PETSC_EXTERN void petscquadratureview_(PetscQuadrature *q, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscQuadratureView(*q, v);
}
