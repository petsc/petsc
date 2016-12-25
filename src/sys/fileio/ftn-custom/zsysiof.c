#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbinaryopen_           PETSCBINARYOPEN
#define petsctestfile_             PETSCTESTFILE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbinaryopen_           petscbinaryopen
#define petsctestfile_             petsctestfile
#endif

PETSC_EXTERN void PETSC_STDCALL petscbinaryopen_(char* name PETSC_MIXED_LEN(len),PetscFileMode *type,int *fd,
                                    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name,len,c1);
  *ierr = PetscBinaryOpen(c1,*type,fd);
  FREECHAR(name,c1);
}

PETSC_EXTERN void PETSC_STDCALL petsctestfile_(char* name PETSC_MIXED_LEN(len),char* mode PETSC_MIXED_LEN(len1),PetscBool *flg,PetscErrorCode *ierr PETSC_END_LEN(len) PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(name,len,c1);
  *ierr = PetscTestFile(c1,*mode,flg);
  FREECHAR(name,c1);
}

