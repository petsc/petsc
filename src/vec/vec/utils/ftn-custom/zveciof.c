#include "private/fortranimpl.h"
#include "petscvec.h"
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecload_                  VECLOAD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecload_                  vecload
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL vecload_(PetscViewer *viewer,Vec *newvec,PetscErrorCode *ierr PETSC_END_LEN(len))
{ 
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = VecLoad(v,*newvec);
}

EXTERN_C_END
