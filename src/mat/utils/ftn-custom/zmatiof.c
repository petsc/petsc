#include "private/fortranimpl.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matload_                         MATLOAD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matload_                         matload
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matload_(PetscViewer *viewer,CHAR outtype PETSC_MIXED_LEN(len),Mat *newmat,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  PetscViewer v;
  FIXCHAR(outtype,len,t);
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = MatLoad(v,t,newmat);
  FREECHAR(outtype,t);
}

EXTERN_C_END
