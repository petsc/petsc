
#include <petscis.h>
#include <../src/sys/f90-src/f90impl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define isgetindicesf90_           ISGETINDICESF90
#define isrestoreindicesf90_       ISRESTOREINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define isgetindicesf90_           isgetindicesf90
#define isrestoreindicesf90_       isrestoreindicesf90
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL isgetindicesf90_(IS *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscInt       len;

  *__ierr = ISGetIndices(*x,&fa); if (*__ierr) return;
  *__ierr = ISGetLocalSize(*x,&len);   if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*)fa,PETSC_INT,1,len,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL isrestoreindicesf90_(IS *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;

  *__ierr = F90Array1dAccess(ptr,PETSC_INT,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = ISRestoreIndices(*x,&fa);
}

EXTERN_C_END




