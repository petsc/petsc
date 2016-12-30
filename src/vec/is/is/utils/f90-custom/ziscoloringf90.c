
#include <petscis.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define iscoloringgetisf90_        ISCOLORINGGETISF90
#define iscoloringrestoreisf90_    ISCOLORINGRESTOREISF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define iscoloringgetisf90_        iscoloringgetisf90
#define iscoloringrestoreisf90_    iscoloringrestoreisf90
#endif

PETSC_EXTERN void PETSC_STDCALL iscoloringgetisf90_(ISColoring *iscoloring,PetscInt *n,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  IS               *lis;
  PetscFortranAddr *newisint;
  int              i;

  *__ierr = ISColoringGetIS(*iscoloring,n,&lis); if (*__ierr) return;
  *__ierr = PetscMalloc1(*n,&newisint); if (*__ierr) return;
  for (i=0; i<*n; i++) newisint[i] = (PetscFortranAddr)lis[i];
  *__ierr = F90Array1dCreate(newisint,PETSC_FORTRANADDR,1,*n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void PETSC_STDCALL iscoloringrestoreisf90_(ISColoring *iscoloring,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFortranAddr *is;

  *__ierr = F90Array1dAccess(ptr,PETSC_FORTRANADDR,(void**)&is PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,PETSC_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = ISColoringRestoreIS(*iscoloring,(IS**)is);if (*__ierr) return;
  *__ierr = PetscFree(is);
}




