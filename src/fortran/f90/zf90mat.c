/*$Id: zf90mat.c,v 1.9 2000/09/07 04:51:11 balay Exp balay $*/

#include "petscmat.h"
#include "petscf90.h"

#if defined (PETSC_HAVE_F90_H)

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matgetarrayf90_            MATGETARRAYF90
#define matrestorearrayf90_        MATRESTOREARRAYF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matgetarrayf90_            matgetarrayf90
#define matrestorearrayf90_        matrestorearrayf90
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matgetarrayf90_(Mat *mat,F90Array2d *ptr,int *__ierr)
{
  Scalar *fa;
  int    m,n;
  *__ierr = MatGetArray(*mat,&fa);       if (*__ierr) return;
  *__ierr = MatGetLocalSize(*mat,&m,&n); if (*__ierr) return;
  *__ierr = F90Array2dCreate(fa,PETSC_SCALAR,1,m,1,n,ptr);
}
void PETSC_STDCALL matrestorearrayf90_(Mat *mat,F90Array2d *ptr,int *__ierr)
{
  Scalar *fa;
  *__ierr = F90Array2dAccess(ptr,(void **)&fa);if (*__ierr) return;
  *__ierr = F90Array2dDestroy(ptr);if (*__ierr) return;
  *__ierr = MatRestoreArray(*mat,&fa);
}
EXTERN_C_END

#else  /* !defined (PETSC_HAVE_F90_H) */

/*
     Dummy function so that compilers won't complain about 
  empty files.
*/
int F90mat_ZF90_Dummy(int dummy)
{
  return 0;
}
 

#endif



