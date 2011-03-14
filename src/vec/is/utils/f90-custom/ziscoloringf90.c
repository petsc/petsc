
#include <petscis.h>
#include <../src/sys/f90-src/f90impl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define iscoloringgetisf90_        ISCOLORINGGETISF90
#define iscoloringrestoreisf90_    ISCOLORINGRESTOREISF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define iscoloringgetisf90_        iscoloringgetisf90
#define iscoloringrestoreisf90_    iscoloringrestoreisf90
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL iscoloringgetisf90_(ISColoring *iscoloring,PetscInt *n,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  IS *lis;
  PetscFortranAddr *newisint;
  int i;
  *__ierr  = ISColoringGetIS(*iscoloring,n,&lis); if (*__ierr) return;
  *__ierr = PetscMalloc((*n)*sizeof(PetscFortranAddr),&newisint); if (*__ierr) return;
  for (i=0; i<*n; i++) {
    newisint[i] = (PetscFortranAddr)lis[i];
  }
  *__ierr = F90Array1dCreate(newisint,PETSC_FORTRANADDR,1,*n,ptr PETSC_F90_2PTR_PARAM(ptrd));
}

void PETSC_STDCALL iscoloringrestoreisf90_(ISColoring *iscoloring,F90Array1d *ptr,int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscFortranAddr *is;

  *__ierr = F90Array1dAccess(ptr,PETSC_FORTRANADDR,(void**)&is PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,PETSC_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = ISColoringRestoreIS(*iscoloring,(IS **)is);if (*__ierr) return;
  *__ierr = PetscFree(is);
}

EXTERN_C_END




