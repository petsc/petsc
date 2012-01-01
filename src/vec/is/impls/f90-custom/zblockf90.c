
#include <petscis.h>
#include <../src/sys/f90-src/f90impl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define isblockgetindicesf90_      ISBLOCKGETINDICESF90
#define isblockrestoreindicesf90_  ISBLOCKRESTOREINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define isblockgetindicesf90_      isblockgetindicesf90
#define isblockrestoreindicesf90_  isblockrestoreindicesf90
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL isblockgetindicesf90_(IS *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscInt  len;
  *__ierr = ISBlockGetIndices(*x,&fa);      if (*__ierr) return;
  *__ierr = ISBlockGetLocalSize(*x,&len);        if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*)fa,PETSC_INT,1,len,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
void PETSC_STDCALL isblockrestoreindicesf90_(IS *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  *__ierr = F90Array1dAccess(ptr,PETSC_INT,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,PETSC_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = ISBlockRestoreIndices(*x,&fa);
}
EXTERN_C_END




