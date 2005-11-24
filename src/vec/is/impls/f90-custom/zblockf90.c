
#include "petscis.h"
#include "src/sys/f90/f90impl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define isblockgetindicesf90_      ISBLOCKGETINDICESF90
#define isblockrestoreindicesf90_  ISBLOCKRESTOREINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define isblockgetindicesf90_      isblockgetindicesf90
#define isblockrestoreindicesf90_  isblockrestoreindicesf90
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL isblockgetindicesf90_(IS *x,F90Array1d *ptr,int *__ierr)
{
  int    *fa;
  int    len;
  *__ierr = ISBlockGetIndices(*x,&fa);      if (*__ierr) return;
  *__ierr = ISBlockGetSize(*x,&len);        if (*__ierr) return;
  *__ierr = F90Array1dCreate(fa,PETSC_INT,1,len,ptr);
}
void PETSC_STDCALL isblockrestoreindicesf90_(IS *x,F90Array1d *ptr,int *__ierr)
{
  int    *fa;
  *__ierr = F90Array1dAccess(ptr,(void**)&fa);if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr);if (*__ierr) return;
  *__ierr = ISBlockRestoreIndices(*x,&fa);
}
EXTERN_C_END




