#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tsroswsettype_                   TSROSWSETTYPE
#define tsroswgettype_                   TSROSWGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsroswsettype_                   tsroswsettype
#define tsroswgettype_                   tsroswgettype
#endif

PETSC_EXTERN void tsroswsettype_(TS *ts,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSRosWSetType(*ts,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void tsroswgettype_(TS *ts,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = TSRosWGetType(*ts,&tname);
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}
