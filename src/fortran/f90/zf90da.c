/*$Id: zf90da.c,v 1.5 2000/01/11 21:03:54 bsmith Exp balay $*/

#include "src/fortran/f90/zf90.h"
#include "petscda.h"

#if !defined (PETSC_HAVE_NOF90)

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dagetglobalindicesf90_     DAGETGLOBALINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetglobalindicesf90_     dagetglobalindicesf90
#endif

EXTERN_C_BEGIN
void dagetglobalindicesf90_(DA da,int *n,array1d *indices,int *__ierr)
{
  int *idx;
  *__ierr = DAGetGlobalIndices((DA)PetscToPointer(da),n,&idx);
  if (*__ierr) return;
  *__ierr = PetscF90Create1dArrayInt(idx,*n,indices);
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



