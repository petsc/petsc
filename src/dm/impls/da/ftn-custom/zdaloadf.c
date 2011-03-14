
#include <private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdaload_                      DMDALOAD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdaload_                      dmdaload
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dmdaload_(PetscViewer *viewer,PetscInt *M,PetscInt *N,PetscInt *P,DM *da,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = DMDALoad(v,*M,*N,*P,da);
}
EXTERN_C_END
