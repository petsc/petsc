#include "private/fortranimpl.h"
#include "petscvec.h"
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecload_                  VECLOAD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecload_                  vecload
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL vecload_(PetscViewer *viewer,CHAR outtype PETSC_MIXED_LEN(len),Vec *newvec,PetscErrorCode *ierr PETSC_END_LEN(len))
{ 
  char *t;
  PetscViewer v;
  FIXCHAR(outtype,len,t);
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = VecLoad(v,t,newvec);
}

EXTERN_C_END
