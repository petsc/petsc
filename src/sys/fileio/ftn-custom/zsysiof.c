#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbinaryopen_            PETSCBINARYOPEN
#define petsctestfile_              PETSCTESTFILE
#define petscbinaryreadint_         PETSCBINARYREADINT
#define petscbinaryreadreal_        PETSCBINARYREADREAL
#define petscbinaryreadcomplex_     PETSCBINARYREADCOMPLEX
#define petscbinaryreadrealcnt_     PETSCBINARYREADREALCNT
#define petscbinaryreadcomplexcnt_  PETSCBINARYREADCOMPLEXCNT
#define petscbinaryreadintcnt_      PETSCBINARYREADINTCNT
#define petscbinaryreadint1_        PETSCBINARYREADINT1
#define petscbinaryreadreal1_       PETSCBINARYREADREAL1
#define petscbinaryreadcomplex1_    PETSCBINARYREADCOMPLEX1
#define petscbinaryreadint1cnt_     PETSCBINARYREADINT1CNT
#define petscbinaryreadreal1cnt_    PETSCBINARYREADREAL1CNT
#define petscbinaryreadcomplex1cnt_ PETSCBINARYREADCOMPLEX1CNT
#define petscbinarywriteint_        PETSCBINARYWRITEINT
#define petscbinarywritereal_       PETSCBINARYWRITEREAL
#define petscbinarywritecomplex_    PETSCBINARYWRITECOMPLEX
#define petscbinarywriteint1_       PETSCBINARYWRITEINT1
#define petscbinarywritereal1_      PETSCBINARYWRITEREAL1
#define petscbinarywritecomplex1_   PETSCBINARYWRITECOMPLEX1
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbinaryopen_            petscbinaryopen
#define petsctestfile_              petsctestfile
#define petscbinaryreadint_         petscbinaryreadint
#define petscbinaryreadreal_        petscbinaryreadreal
#define petscbinaryreadcomplex_     petscbinaryreadcomplex
#define petscbinaryreadintcnt_      petscbinaryreadintcnt
#define petscbinaryreadrealcnt_     petscbinaryreadrealcnt
#define petscbinaryreadcomplexcnt_  petscbinaryreadcomplexcnt
#define petscbinaryreadint1_        petscbinaryreadint1
#define petscbinaryreadrea11_       petscbinaryreadrea11
#define petscbinaryreadcomplex1_    petscbinaryreadcomplex1
#define petscbinaryreadint1cnt_     petscbinaryreadint1cnt
#define petscbinaryreadreal1cnt_    petscbinaryreadreal1cnt
#define petscbinaryreadcomplex1cnt_ petscbinaryreadcomplex1cnt
#define petscbinarywriteint_        petscbinarywriteint
#define petscbinarywritereal_       petscbinarywritereal
#define petscbinarywritecomplex_    petscbinarywritecomplex
#define petscbinarywriteint1_       petscbinarywriteint1
#define petscbinarywritereal1_      petscbinarywritereal1
#define petscbinarywritecomplex1_   petscbinarywritecomplex1
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

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadintcnt_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadrealcnt_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadcomplexcnt_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadint1cnt_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadreal1cnt_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
{
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscBinaryRead(*fd,data,*num,count,*type);if (*ierr) return;
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryreadcomplex1cnt_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
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

#if defined(__cplusplus)
}
#endif
