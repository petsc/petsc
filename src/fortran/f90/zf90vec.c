/*$Id: zf90vec.c,v 1.8 2000/05/24 22:17:56 balay Exp bsmith $*/

#include "src/fortran/f90/zf90.h"
#include "petscis.h"
#include "petscvec.h"

#if !defined (PETSC_HAVE_NOF90)

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

void isgetindicesf90_(IS *x,array1d *ptr,int *__ierr)
{
  int    *fa;
  int    len;
  *__ierr = ISGetIndices(*x,&fa); if (*__ierr) return;
  *__ierr = ISGetLocalSize(*x,&len);   if (*__ierr) return;
  *__ierr = PetscF90Create1dArrayInt(fa,len,ptr);
}
void isrestoreindicesf90_(IS *x,array1d *ptr,int *__ierr)
{
  int    *fa;
  *__ierr = PetscF90Get1dArrayInt(ptr,&fa);if (*__ierr) return;
  *__ierr = PetscF90Destroy1dArrayInt(ptr);if (*__ierr) return;
  *__ierr = ISRestoreIndices(*x,&fa);
}

void isblockgetindicesf90_(IS *x,array1d *ptr,int *__ierr)
{
  int    *fa;
  int    len;
  *__ierr = ISBlockGetIndices(*x,&fa);      if (*__ierr) return;
  *__ierr = ISBlockGetSize(*x,&len);        if (*__ierr) return;
  *__ierr = PetscF90Create1dArrayInt(fa,len,ptr);
}
void isblockrestoreindicesf90_(IS *x,array1d *ptr,int *__ierr)
{
  int    *fa;
  *__ierr = PetscF90Get1dArrayInt(ptr,&fa);if (*__ierr) return;
  *__ierr = PetscF90Destroy1dArrayInt(ptr);if (*__ierr) return;
  *__ierr = ISBlockRestoreIndices(*x,&fa);
}


void iscoloringgetisf90_(ISColoring *iscoloring,int *n,array1d *ptr,int *__ierr)
{
  IS *lis;
  PetscFortranAddr *newisint;
  int i;
  *__ierr  = ISColoringGetIS(*iscoloring,n,&lis); if (*__ierr) return;
  newisint = (PetscFortranAddr*)PetscMalloc((*n)*sizeof(PetscFortranAddr)); 
  if (!newisint) {*__ierr = PETSC_ERR_MEM; return;}
  for (i=0; i<*n; i++) {
    newisint[i] = (PetscFortranAddr)lis[i];
  }
  *__ierr = PetscF90Create1dArrayPetscFortranAddr(newisint,*n,ptr);
}

void iscoloringrestoreisf90__(ISColoring *iscoloring,array1d *ptr,int *__ierr)
{
  PetscFortranAddr *is;

  *__ierr = PetscF90Get1dArrayPetscFortranAddr(ptr,&is);if (*__ierr) return;
  *__ierr = PetscF90Destroy1dArrayPetscFortranAddr(ptr);if (*__ierr) return;
  *__ierr = ISColoringRestoreIS(*iscoloring,is);if (*__ierr) return;
  *__ierr = PetscFree(is);
}

/* ---------------------------------------------------------------*/

void vecgetarrayf90_(Vec *x,array1d *ptr,int *__ierr)
{
  Scalar *fa;
  int    len;
  *__ierr = VecGetArray(*x,&fa);      if (*__ierr) return;
  *__ierr = VecGetLocalSize(*x,&len); if (*__ierr) return;
  *__ierr = PetscF90Create1dArrayScalar(fa,len,ptr);
}
void vecrestorearrayf90_(Vec *x,array1d *ptr,int *__ierr)
{
  Scalar *fa;
  *__ierr = PetscF90Get1dArrayScalar(ptr,&fa);if (*__ierr) return;
  *__ierr = PetscF90Destroy1dArrayScalar(ptr);if (*__ierr) return;
  *__ierr = VecRestoreArray(*x,&fa);
}

void vecduplicatevecsf90_(Vec *v,int *m,array1d *ptr,int *__ierr)
{
  Vec *lV;
  PetscFortranAddr *newvint;
  int i;
  *__ierr = VecDuplicateVecs(*v,*m,&lV); if (*__ierr) return;
  newvint = (PetscFortranAddr*)PetscMalloc((*m)*sizeof(PetscFortranAddr)); 
  if (!newvint) {*__ierr = PETSC_ERR_MEM; return;}
  for (i=0; i<*m; i++) {
    newvint[i] = (PetscFortranAddr)lV[i];
  }
  *__ierr = PetscFree(lV); if (*__ierr) return;
  *__ierr = PetscF90Create1dArrayPetscFortranAddr(newvint,*m,ptr);
}

void vecdestroyvecsf90_(array1d *ptr,int *m,int *__ierr)
{
  PetscFortranAddr *vecs;
  int       i;

  *__ierr = PetscF90Get1dArrayPetscFortranAddr(ptr,&vecs);if (*__ierr) return;
  for (i=0; i<*m; i++) {
    *__ierr = VecDestroy(&vecs[i]);
    if (*__ierr) return;
  }
  *__ierr = PetscF90Destroy1dArrayPetscFortranAddr(ptr);if (*__ierr) return;
  *__ierr = PetscFree(vecs);
}

EXTERN_C_END

#else  /* !defined (PETSC_HAVE_NOF90) */


/*
     Dummy function so that compilers won't complain about 
  empty files.
*/
int F90vec_ZF90_Dummy(int dummy)
{
  return 0;
}
 

#endif



