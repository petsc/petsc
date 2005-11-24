
#include "petscis.h"
#include "src/sys/f90/f90impl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define isgetindicesf90_           ISGETINDICESF90
#define isrestoreindicesf90_       ISRESTOREINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define isgetindicesf90_           isgetindicesf90
#define isrestoreindicesf90_       isrestoreindicesf90
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL isgetindicesf90_(IS *x,F90Array1d *ptr,int *__ierr)
{
  int    *fa;
  int    len;

  *__ierr = ISGetIndices(*x,&fa); if (*__ierr) return;
  *__ierr = ISGetLocalSize(*x,&len);   if (*__ierr) return;
  *__ierr = F90Array1dCreate(fa,PETSC_INT,1,len,ptr);
}
void PETSC_STDCALL isrestoreindicesf90_(IS *x,F90Array1d *ptr,int *__ierr)
{
  int    *fa;
  *__ierr = F90Array1dAccess(ptr,(void**)&fa);if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr);if (*__ierr) return;
  *__ierr = ISRestoreIndices(*x,&fa);
}

EXTERN_C_END




