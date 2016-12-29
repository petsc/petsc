#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscgetarchtype_                  PETSCGETARCHTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscgetarchtype_                  petscgetarchtype
#endif

PETSC_EXTERN void PETSC_STDCALL petscgetarchtype_(char* str PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *tstr;
  size_t tlen;
  tstr  = str;
  tlen  = len; /* int to size_t */
  *ierr = PetscGetArchType(tstr,tlen);
  FIXRETURNCHAR(PETSC_TRUE,str,len);

}

