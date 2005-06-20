
#include "zpetsc.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dagetmatrix_                 DAGETMATRIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetmatrix_                 dagetmatrix
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dagetmatrix_(DA *da,CHAR mat_type PETSC_MIXED_LEN(len),Mat *J,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(mat_type,len,t);
  *ierr = DAGetMatrix(*da,t,J);
  FREECHAR(mat_type,t);
}

EXTERN_C_END
