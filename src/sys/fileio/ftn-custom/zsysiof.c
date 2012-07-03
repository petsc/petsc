#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbinaryopen_           PETSCBINARYOPEN
#define petsctestfile_             PETSCTESTFILE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbinaryopen_           petscbinaryopen
#define petsctestfile_             petsctestfile
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscbinaryopen_(CHAR name PETSC_MIXED_LEN(len),PetscFileMode *type,int *fd,
                                    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name,len,c1);
  *ierr = PetscBinaryOpen(c1,*type,fd);
  FREECHAR(name,c1);
}

void PETSC_STDCALL petsctestfile_(CHAR name PETSC_MIXED_LEN(len),CHAR mode PETSC_MIXED_LEN(len1),PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len) PETSC_END_LEN(len1))
{
  char *c1,*m1;

  FIXCHAR(name,len,c1);
  FIXCHAR(mode,len1,m1);
  *ierr = PetscTestFile(c1,*m1,flg);
  FREECHAR(name,c1);
  FREECHAR(mode,m1);
}

EXTERN_C_END
