#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tssettype_                           TSSETTYPE
#define tsgettype_                           TSGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tssettype_                           tssettype
#define tsgettype_                           tsgettype
#endif

PETSC_EXTERN void PETSC_STDCALL tssettype_(TS *ts,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSSetType(*ts,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL tsgettype_(TS *ts,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = TSGetType(*ts,&tname);
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

