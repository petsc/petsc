
#include <petscdmda.h>
#include <../src/sys/f90-src/f90impl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmdagetglobalindicesf90_     DMDAGETGLOBALINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdagetglobalindicesf90_     dmdagetglobalindicesf90
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dmdagetglobalindicesf90_(DM *da,PetscInt *n,F90Array1d *indices,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt *idx;
  *ierr = DMDAGetGlobalIndices(*da,n,&idx); if (*ierr) return;
  *ierr = F90Array1dCreate(idx,PETSC_INT,1,*n,indices PETSC_F90_2PTR_PARAM(ptrd));
}
EXTERN_C_END



