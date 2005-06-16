#include "zpetsc.h"
#include "petsc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscstrncpy_              PETSCSTRNCPY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscstrncpy_              petscstrncpy
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscstrncpy_(CHAR s1 PETSC_MIXED_LEN(len1),CHAR s2 PETSC_MIXED_LEN(len2),int *n,
                                 PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;
  int  m;

#if defined(PETSC_USES_CPTOFCD)
  t1 = _fcdtocp(s1); 
  t2 = _fcdtocp(s2); 
  m = *n; if (_fcdlen(s1) < m) m = _fcdlen(s1); if (_fcdlen(s2) < m) m = _fcdlen(s2);
#else
  t1 = s1;
  t2 = s2;
  m = *n; if (len1 < m) m = len1; if (len2 < m) m = len2;
#endif
  *ierr = PetscStrncpy(t1,t2,m);
}

EXTERN_C_END
