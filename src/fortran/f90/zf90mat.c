/*$Id: zf90mat.c,v 1.3 1999/05/12 03:34:46 bsmith Exp bsmith $*/

#include "src/fortran/f90/zf90.h"
#include "mat.h"

#if !defined (PETSC_HAVE_NOF90)

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matgetarrayf90_            MATGETARRAYF90
#define matrestorearrayf90_        MATRESTOREARRAYF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matgetarrayf90_            matgetarrayf90
#define matrestorearrayf90_        matrestorearrayf90
#endif

EXTERN_C_BEGIN
void matgetarrayf90_(Vec x,array2d *ptr,int *__ierr)
{
  Scalar *fa;
  int    m,n;
  Mat    xin = (Mat)PetscToPointer(x);
  *__ierr = MatGetArray(xin,&fa);      if (*__ierr) return;
  *__ierr = MatGetLocalSize(xin,&m,&n); if (*__ierr) return;
  *__ierr = PetscF90Create2dArrayScalar(fa,m,n,ptr);
}
void matrestorearrayf90_(Vec x,array2d *ptr,int *__ierr)
{
  Scalar *fa;
  Mat    xin = (Mat)PetscToPointer(x);
  *__ierr = PetscF90Get2dArrayScalar(ptr,&fa);if (*__ierr) return;
  *__ierr = PetscF90Destroy2dArrayScalar(ptr);if (*__ierr) return;
  *__ierr = MatRestoreArray(xin,&fa);         if (*__ierr) return;
}
EXTERN_C_END

#else  /* !defined (PETSC_HAVE_NOF90) */

/*
     Dummy function so that compilers won't complain about 
  empty files.
*/
int F90mat_ZF90_Dummy(int dummy)
{
  return 0;
}
 

#endif



