#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscgethostname_                  PETSCGETHOSTNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscgethostname_                  petscgethostname
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscgethostname_(CHAR str PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tstr;
  size_t tlen;
  tstr = str;
  tlen = len; /* int to size_t */
  *ierr = PetscGetHostName(tstr,tlen);
  FIXRETURNCHAR(PETSC_TRUE,str,len);

}

EXTERN_C_END
