#include "zpetsc.h"
#include "petscpc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcsettype_                 PCSETTYPE
#define pcgettype_                 PCGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcsettype_                 pcsettype
#define pcgettype_                 pcgettype
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL pcsettype_(PC *pc,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PCSetType(*pc,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL pcgettype_(PC *pc,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PCGetType(*pc,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  *ierr = PetscStrncpy(t,tname,len1); if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
#endif
  FIXRETURNCHAR(name,len);

}

EXTERN_C_END
