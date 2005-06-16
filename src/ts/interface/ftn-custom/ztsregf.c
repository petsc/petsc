#include "zpetsc.h"
#include "petscts.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tssettype_                           TSSETTYPE
#define tsgettype_                           TSGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tssettype_                           tssettype
#define tsgettype_                           tsgettype
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL tssettype_(TS *ts,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSSetType(*ts,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL tsgettype_(TS *ts,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = TSGetType(*ts,(TSType *)&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *ierr = PetscStrncpy(name,tname,len);
#endif
  FIXRETURNCHAR(name,len);
}


EXTERN_C_END
