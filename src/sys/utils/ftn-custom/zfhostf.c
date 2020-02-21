#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscgethostname_                  PETSCGETHOSTNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscgethostname_                  petscgethostname
#endif

PETSC_EXTERN void petscgethostname_(char* str,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char   *tstr;
  size_t tlen;
  tstr  = str;
  tlen  = len; /* int to size_t */
  *ierr = PetscGetHostName(tstr,tlen);
  FIXRETURNCHAR(PETSC_TRUE,str,len);
}

