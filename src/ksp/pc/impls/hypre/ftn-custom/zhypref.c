#include <petsc-private/fortranimpl.h>
#include <petscpc.h>

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
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}

EXTERN_C_END
