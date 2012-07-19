#include <petsc-private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matnullspaceview_                MATNULLSPACEVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matnullspaceview_                matnullspaceview
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matnullspaceview_(MatNullSpace *sp,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = MatNullSpaceView(*sp,v);
}
EXTERN_C_END
