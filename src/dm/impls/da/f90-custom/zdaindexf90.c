
#include <petscdmda.h>
#include <../src/sys/f90-src/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdagetglobalindicesf90_     DMDAGETGLOBALINDICESF90
#define dmdarestoreglobalindicesf90_ DMDARESTOREGLOBALINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdagetglobalindicesf90_     dmdagetglobalindicesf90
#define dmdarestoreglobalindicesf90_ dmdarestoreglobalindicesf90
#endif

PETSC_EXTERN void PETSC_STDCALL dmdagetglobalindicesf90_(DM *da,PetscInt *n,F90Array1d *indices,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;
  *ierr = DMDAGetGlobalIndices(*da,n,&idx); if (*ierr) return;
  *ierr = F90Array1dCreate((void*)idx,PETSC_INT,1,*n,indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL dmdarestoreglobalindicesf90_(DM *da,PetscInt *n,F90Array1d *ptr,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;

  *ierr = F90Array1dAccess(ptr,PETSC_INT,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr,PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*ierr) return;
  *ierr = DMDARestoreGlobalIndices(*da,n,&fa); if (*ierr) return;
}



