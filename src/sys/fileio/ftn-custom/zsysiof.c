#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbinaryopen_           PETSCBINARYOPEN
#define petscbinaryread_           PETSCBINARYREAD
#define petsctestfile_             PETSCTESTFILE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbinaryopen_           petscbinaryopen
#define petscbinaryread_           petscbinaryread
#define petsctestfile_             petsctestfile
#endif

PETSC_EXTERN void PETSC_STDCALL petscbinaryopen_(char* name PETSC_MIXED_LEN(len),PetscFileMode *type,int *fd,
                                    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name,len,c1);
  *ierr = PetscBinaryOpen(c1,*type,fd);if (*ierr) return;
  FREECHAR(name,c1);
}

PETSC_EXTERN void PETSC_STDCALL  petscbinaryread_(int *fd,void *data,PetscInt *num,PetscInt *count,PetscDataType *type,int *ierr)
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
