
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define daload_                      DALOAD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define daload_                      daload
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL daload_(PetscViewer *viewer,PetscInt *M,PetscInt *N,PetscInt *P,DA *da,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = DALoad(v,*M,*N,*P,da);
}
EXTERN_C_END
