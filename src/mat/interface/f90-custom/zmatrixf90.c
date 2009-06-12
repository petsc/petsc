
#include "petscmat.h"
#include "../src/sys/f90-src/f90impl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define matgetarrayf90_            MATGETARRAYF90
#define matrestorearrayf90_        MATRESTOREARRAYF90
#define matgetghostsf90_           MATGETGHOSTSF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matgetarrayf90_            matgetarrayf90
#define matrestorearrayf90_        matrestorearrayf90
#define matgetghostsf90_           matgetghostsf90
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL matgetghostsf90_(Mat *mat,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *ghosts;
  PetscInt       N;

  *ierr = MatGetGhosts(*mat,&N,&ghosts); if (*ierr) return;
  *ierr = F90Array1dCreate((PetscInt *)ghosts,PETSC_INT,1,N,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL matgetarrayf90_(Mat *mat,F90Array2d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt     m,n;
  *ierr = MatGetArray(*mat,&fa);       if (*ierr) return;
  *ierr = MatGetLocalSize(*mat,&m,&n); if (*ierr) return;
  *ierr = F90Array2dCreate(fa,PETSC_SCALAR,1,m,1,n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL matrestorearrayf90_(Mat *mat,F90Array2d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  *ierr = F90Array2dAccess(ptr,PETSC_SCALAR,(void **)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = F90Array2dDestroy(ptr,PETSC_SCALAR PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = MatRestoreArray(*mat,&fa);
}
EXTERN_C_END



