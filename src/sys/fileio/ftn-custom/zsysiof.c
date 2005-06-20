#include "zpetsc.h"
#include "petscsys.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbinaryopen_           PETSCBINARYOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbinaryopen_           petscbinaryopen
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscbinaryopen_(CHAR name PETSC_MIXED_LEN(len),PetscViewerFileType *type,int *fd,
                                    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name,len,c1);
  *ierr = PetscBinaryOpen(c1,*type,fd);
  FREECHAR(name,c1);
}

EXTERN_C_END
