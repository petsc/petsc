/*$Id: zf90da.c,v 1.10 2000/09/06 23:01:04 balay Exp balay $*/

#include "petscda.h"
#include "petscf90.h"

#if defined (PETSC_HAVE_F90_H)

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

#else  /* !defined (PETSC_HAVE_F90_H) */

/*
     Dummy function so that compilers won't complain about 
  empty files.
*/
int F90da_ZF90_Dummy(int dummy)
{
  return 0;
}

#endif



