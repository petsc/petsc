/*$Id: zf90da.c,v 1.9 2000/07/11 20:59:59 balay Exp balay $*/

#include "petscda.h"
#include "petscf90.h"

#if !defined (PETSC_HAVE_NOF90)

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dagetglobalindicesf90_     DAGETGLOBALINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetglobalindicesf90_     dagetglobalindicesf90
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dagetglobalindicesf90_(DA *da,int *n,F90Array1d indices,int *__ierr)
{
  int *idx;
  *__ierr = DAGetGlobalIndices(*da,n,&idx); if (*__ierr) return;
  *__ierr = F90Array1dCreate(idx,PETSC_INT,1,*n,indices);
}
EXTERN_C_END

#else  /* !defined (PETSC_HAVE_NOF90) */

/*
     Dummy function so that compilers won't complain about 
  empty files.
*/
int F90da_ZF90_Dummy(int dummy)
{
  return 0;
}

#endif



