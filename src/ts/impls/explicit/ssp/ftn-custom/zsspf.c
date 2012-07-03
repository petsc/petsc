#include <petsc-private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tssspsettype_                   TSSSPSETTYPE
#define tssspgettype_                   TSSSPGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tssspsettype_                   tssspsettype
#define tssspgettype_                   tssspgettype
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL tssspsettype_(TS *ts,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSSSPSetType(*ts,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL tssspgettype_(TS *ts,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = TSSSPGetType(*ts,&tname);
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

EXTERN_C_END
