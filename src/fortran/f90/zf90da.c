/*$Id: zf90da.c,v 1.12 2000/09/22 18:53:58 balay Exp bsmith $*/

#include "petscda.h"
#include "petscf90.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dagetglobalindicesf90_     DAGETGLOBALINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetglobalindicesf90_     dagetglobalindicesf90
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dagetglobalindicesf90_(DA *da,int *n,F90Array1d *indices,int *__ierr)
{
  int *idx;
  *__ierr = DAGetGlobalIndices(*da,n,&idx); if (*__ierr) return;
  *__ierr = F90Array1dCreate(idx,PETSC_INT,1,*n,indices);
}
EXTERN_C_END



