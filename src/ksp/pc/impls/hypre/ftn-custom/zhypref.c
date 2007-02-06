#include "zpetsc.h"
#include "petscpc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pchypresettype_            PCHYPRESETTYPE
#define pchypregettype_            PCHYPREGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pchypresettype_            pchypresettype
#define pchypregettype_            pchypregettype
#endif


EXTERN_C_BEGIN

void PETSC_STDCALL  pchypresettype_(PC *pc, CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = PCHYPRESetType(*pc,t);
  FREECHAR(name,t);
}

void PETSC_STDCALL pchypregettype_(PC *pc,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PCHYPREGetType(*pc,&tname);
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
