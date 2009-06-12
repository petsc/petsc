
#include "petscda.h"
#include "../src/sys/f90-src/f90impl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dagetglobalindicesf90_     DAGETGLOBALINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetglobalindicesf90_     dagetglobalindicesf90
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dagetglobalindicesf90_(DA *da,PetscInt *n,F90Array1d *indices,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *idx;
  *ierr = DAGetGlobalIndices(*da,n,&idx); if (*ierr) return;
  *ierr = F90Array1dCreate(idx,PETSC_INT,1,*n,indices PETSC_F90_2PTR_PARAM(ptrd));
}
EXTERN_C_END



