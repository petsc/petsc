#include "zpetsc.h"
#include "petscsys.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscgetarchtype_                  PETSCGETARCHTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscgetarchtype_                  petscgetarchtype
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscgetarchtype_(CHAR str PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USES_CPTOFCD)
  char *tstr = _fcdtocp(str); 
  int  len1 = _fcdlen(str);
  *ierr = PetscGetArchType(tstr,len1);
#else
  *ierr = PetscGetArchType(str,len);
#endif
  FIXRETURNCHAR(str,len);

}


EXTERN_C_END
