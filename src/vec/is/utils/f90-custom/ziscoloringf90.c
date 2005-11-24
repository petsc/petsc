
#include "petscis.h"
#include "src/sys/f90/f90impl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define iscoloringgetisf90_        ISCOLORINGGETISF90
#define iscoloringrestoreisf90_    ISCOLORINGRESTOREF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define iscoloringgetisf90_        iscoloringgetisf90
#define iscoloringrestoreisf90_    iscoloringrestoreisf90
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL iscoloringgetisf90_(ISColoring *iscoloring,int *n,F90Array1d *ptr,int *__ierr)
{
  IS *lis;
  PetscFortranAddr *newisint;
  int i;
  *__ierr  = ISColoringGetIS(*iscoloring,n,&lis); if (*__ierr) return;
  *__ierr = PetscMalloc((*n)*sizeof(PetscFortranAddr),&newisint); if (*__ierr) return;
  for (i=0; i<*n; i++) {
    newisint[i] = (PetscFortranAddr)lis[i];
  }
  *__ierr = F90Array1dCreate(newisint,PETSC_FORTRANADDR,1,*n,ptr);
}

void PETSC_STDCALL iscoloringrestoreisf90_(ISColoring *iscoloring,F90Array1d *ptr,int *__ierr)
{
  PetscFortranAddr *is;

  *__ierr = F90Array1dAccess(ptr,(void**)&is);if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr);if (*__ierr) return;
  *__ierr = ISColoringRestoreIS(*iscoloring,(IS **)is);if (*__ierr) return;
  *__ierr = PetscFree(is);
}

EXTERN_C_END




