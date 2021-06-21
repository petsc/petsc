
#include <petscis.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define isgetindicesf90_           ISGETINDICESF90
#define isrestoreindicesf90_       ISRESTOREINDICESF90
#define isdestroy_                 ISDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define isgetindicesf90_           isgetindicesf90
#define isrestoreindicesf90_       isrestoreindicesf90
#define isdestroy_                 isdestroy
#endif

PETSC_EXTERN void isgetindicesf90_(IS *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscInt       len;

  *__ierr = ISGetIndices(*x,&fa); if (*__ierr) return;
  *__ierr = ISGetLocalSize(*x,&len);   if (*__ierr) return;
  *__ierr = F90Array1dCreate((void*)fa,MPIU_INT,1,len,ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void isrestoreindicesf90_(IS *x,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;

  *__ierr = F90Array1dAccess(ptr,MPIU_INT,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = ISRestoreIndices(*x,&fa);
}

PETSC_EXTERN void isdestroy_(IS *x,int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = ISDestroy(x); if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}

