#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbinaryopen_           PETSCBINARYOPEN
#define petscbinaryread_           PETSCBINARYREAD
#define petsctestfile_             PETSCTESTFILE
#define petscbinarywrite_          PETSCBINARYWRITE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbinarywrite_          petscbinarywrite
#define petscbinaryopen_           petscbinaryopen
#define petscbinaryread_           petscbinaryread
#define petsctestfile_             petsctestfile
#endif

/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif

PETSC_EXTERN void PETSC_STDCALL  petscbinarywriteint_(int *fd,void*p,PetscInt *n,PetscDataType *type,PetscBool  *istemp, int *ierr)
{
  *ierr = PetscBinaryWrite(*fd,p,*n,*type,*istemp);
}

PETSC_EXTERN void PETSC_STDCALL  petscbinarywritereal_(int *fd,void*p,PetscInt *n,PetscDataType *type,PetscBool  *istemp, int *ierr)
{
  *ierr = PetscBinaryWrite(*fd,p,*n,*type,*istemp);
}

PETSC_EXTERN void PETSC_STDCALL  petscbinarywritecomplex_(int *fd,void*p,PetscInt *n,PetscDataType *type,PetscBool  *istemp, int *ierr)
{
  *ierr = PetscBinaryWrite(*fd,p,*n,*type,*istemp);
}

PETSC_EXTERN void PETSC_STDCALL  petscbinarywriteint1_(int *fd,void*p,PetscInt *n,PetscDataType *type,PetscBool  *istemp, int *ierr)
{
  *ierr = PetscBinaryWrite(*fd,p,*n,*type,*istemp);
}

PETSC_EXTERN void PETSC_STDCALL  petscbinarywritereal1_(int *fd,void*p,PetscInt *n,PetscDataType *type,PetscBool  *istemp, int *ierr)
{
  *ierr = PetscBinaryWrite(*fd,p,*n,*type,*istemp);
}

PETSC_EXTERN void PETSC_STDCALL  petscbinarywritecomplex1_(int *fd,void*p,PetscInt *n,PetscDataType *type,PetscBool  *istemp, int *ierr)
{
  *ierr = PetscBinaryWrite(*fd,p,*n,*type,*istemp);
}

PETSC_EXTERN void PETSC_STDCALL petscbinaryopen_(char* name PETSC_MIXED_LEN(len),PetscFileMode *type,int *fd,
                                    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name,len,c1);
  *ierr = PetscBinaryOpen(c1,*type,fd);if (*ierr) return;
  FREECHAR(name,c1);
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadint_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadreal_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadcomplex_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadint1_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadreal1_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadcomplex1_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL petsctestfile_(char* name PETSC_MIXED_LEN(len),char* mode PETSC_MIXED_LEN(len1),PetscBool *flg,PetscErrorCode *ierr PETSC_END_LEN(len) PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(name,len,c1);
  *ierr = PetscTestFile(c1,*mode,flg);if (*ierr) return;
  FREECHAR(name,c1);
}
