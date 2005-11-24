
#include "petscmat.h"
#include "src/sys/f90/f90impl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matgetarrayf90_            MATGETARRAYF90
#define matrestorearrayf90_        MATRESTOREARRAYF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matgetarrayf90_            matgetarrayf90
#define matrestorearrayf90_        matrestorearrayf90
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matgetarrayf90_(Mat *mat,F90Array2d *ptr,int *ierr)
{
  PetscScalar *fa;
  int    m,n;
  *ierr = MatGetArray(*mat,&fa);       if (*ierr) return;
  *ierr = MatGetLocalSize(*mat,&m,&n); if (*ierr) return;
  *ierr = F90Array2dCreate(fa,PETSC_SCALAR,1,m,1,n,ptr);
}
void PETSC_STDCALL matrestorearrayf90_(Mat *mat,F90Array2d *ptr,int *ierr)
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(ptr,(void **)&fa);if (*ierr) return;
  *ierr = F90Array2dDestroy(ptr);if (*ierr) return;
  *ierr = MatRestoreArray(*mat,&fa);
}
EXTERN_C_END



