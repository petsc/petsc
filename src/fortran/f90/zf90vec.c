
#include "petscis.h"
#include "petscvec.h"
#include "src/sys/f90/f90impl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define isgetindicesf90_           ISGETINDICESF90
#define isblockgetindicesf90_      ISBLOCKGETINDICESF90
#define isrestoreindicesf90_       ISRESTOREINDICESF90
#define isblockrestoreindicesf90_  ISBLOCKRESTOREINDICESF90
#define iscoloringgetisf90_        ISCOLORINGGETISF90
#define iscoloringrestoreisf90_    ISCOLORINGRESTOREF90
#define vecgetarrayf90_            VECGETARRAYF90
#define vecrestorearrayf90_        VECRESTOREARRAYF90
#define vecduplicatevecsf90_       VECDUPLICATEVECSF90
#define vecdestroyvecsf90_         VECDESTROYVECSF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define isgetindicesf90_           isgetindicesf90
#define isblockgetindicesf90_      isblockgetindicesf90
#define isrestoreindicesf90_       isrestoreindicesf90 
#define isblockrestoreindicesf90_  isblockrestoreindicesf90
#define iscoloringgetisf90_        iscoloringgetisf90
#define iscoloringrestoreisf90_    iscoloringrestoreisf90
#define vecgetarrayf90_            vecgetarrayf90
#define vecrestorearrayf90_        vecrestorearrayf90
#define vecduplicatevecsf90_       vecduplicatevecsf90
#define vecdestroyvecsf90_         vecdestroyvecsf90
#endif

EXTERN_C_BEGIN

/* --------------------------------------------------------------- */

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

/* ---------------------------------------------------------------*/

void PETSC_STDCALL vecgetarrayf90_(Vec *x,F90Array1d *ptr,int *__ierr)
{
  PetscScalar *fa;
  int    len;
  *__ierr = VecGetArray(*x,&fa);      if (*__ierr) return;
  *__ierr = VecGetLocalSize(*x,&len); if (*__ierr) return;
  *__ierr = F90Array1dCreate(fa,PETSC_SCALAR,1,len,ptr);
}
void PETSC_STDCALL vecrestorearrayf90_(Vec *x,F90Array1d *ptr,int *__ierr)
{
  PetscScalar *fa;
  *__ierr = F90Array1dAccess(ptr,(void**)&fa);if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr);if (*__ierr) return;
  *__ierr = VecRestoreArray(*x,&fa);
}

void PETSC_STDCALL vecduplicatevecsf90_(Vec *v,int *m,F90Array1d *ptr,int *__ierr)
{
  Vec *lV;
  PetscFortranAddr *newvint;
  int i;
  *__ierr = VecDuplicateVecs(*v,*m,&lV); if (*__ierr) return;
  *__ierr = PetscMalloc((*m)*sizeof(PetscFortranAddr),&newvint);  if (*__ierr) return;

  for (i=0; i<*m; i++) {
    newvint[i] = (PetscFortranAddr)lV[i];
  }
  *__ierr = PetscFree(lV); if (*__ierr) return;
  *__ierr = F90Array1dCreate(newvint,PETSC_FORTRANADDR,1,*m,ptr);
}

void PETSC_STDCALL vecdestroyvecsf90_(F90Array1d *ptr,int *m,int *__ierr)
{
  PetscFortranAddr *vecs;
  int       i;

  *__ierr = F90Array1dAccess(ptr,(void**)&vecs);if (*__ierr) return;
  for (i=0; i<*m; i++) {
    *__ierr = VecDestroy((Vec)vecs[i]);
    if (*__ierr) return;
  }
  *__ierr = F90Array1dDestroy(ptr);if (*__ierr) return;
  *__ierr = PetscFree(vecs);
}

EXTERN_C_END




