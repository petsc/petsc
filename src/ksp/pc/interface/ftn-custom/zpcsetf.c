#include <petsc/private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcsettype_                 PCSETTYPE
#define pcgettype_                 PCGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcsettype_                 pcsettype
#define pcgettype_                 pcgettype
#endif

PETSC_EXTERN void PETSC_STDCALL pcsettype_(PC *pc,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PCSetType(*pc,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL pcgettype_(PC *pc,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PCGetType(*pc,&tname);
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

