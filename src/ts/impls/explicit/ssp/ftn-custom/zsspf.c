#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tssspsettype_                   TSSSPSETTYPE
#define tssspgettype_                   TSSSPGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tssspsettype_                   tssspsettype
#define tssspgettype_                   tssspgettype
#endif

PETSC_EXTERN void tssspsettype_(TS *ts,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSSSPSetType(*ts,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void tssspgettype_(TS *ts,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = TSSSPGetType(*ts,&tname);
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}
