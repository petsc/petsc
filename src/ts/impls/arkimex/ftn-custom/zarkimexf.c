#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tsarkimexsettype_                   TSARKIMEXSETTYPE
#define tsarkimexgettype_                   TSARKIMEXGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsarkimexsettype_                   tsarkimexsettype
#define tsarkimexgettype_                   tsarkimexgettype
#endif

PETSC_EXTERN void PETSC_STDCALL tsarkimexsettype_(TS *ts,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSARKIMEXSetType(*ts,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL tsarkimexgettype_(TS *ts,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = TSARKIMEXGetType(*ts,&tname);
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

