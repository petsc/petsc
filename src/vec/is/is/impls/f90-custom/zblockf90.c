
#include <petscis.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define isblockgetindicesf90_      ISBLOCKGETINDICESF90
#define isblockrestoreindicesf90_  ISBLOCKRESTOREINDICESF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define isblockgetindicesf90_      isblockgetindicesf90
#define isblockrestoreindicesf90_  isblockrestoreindicesf90
#endif

PETSC_EXTERN void isblockgetindicesf90_(IS *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscInt       len;
  *__ierr = ISBlockGetIndices(*x,&fa);      if (*__ierr) return;
  *__ierr = ISBlockGetLocalSize(*x,&len);        if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*)fa,MPIU_INT,1,len,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void isblockrestoreindicesf90_(IS *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  *__ierr = F90Array1dAccess(ptr,MPIU_INT,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = ISBlockRestoreIndices(*x,&fa);
}

