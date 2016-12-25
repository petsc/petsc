#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tsroswsettype_                   TSROSWSETTYPE
#define tsroswgettype_                   TSROSWGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsroswsettype_                   tsroswsettype
#define tsroswgettype_                   tsroswgettype
#endif

PETSC_EXTERN void PETSC_STDCALL tsroswsettype_(TS *ts,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSRosWSetType(*ts,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL tsroswgettype_(TS *ts,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = TSRosWGetType(*ts,&tname);
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}
