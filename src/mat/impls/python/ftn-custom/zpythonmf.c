#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matpythonsettype_            MATPYTHONSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matpythonsettype_            matpythonsettype
#endif

PETSC_EXTERN void PETSC_STDCALL matpythonsettype_(Mat *mat, char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = MatPythonSetType(*mat,t);
  FREECHAR(name,t);
}

