#include "private/fortranimpl.h"
#include "petscvec.h"
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecsettype_               VECSETTYPE
#define vecgettype_               VECGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecsettype_               vecsettype
#define vecgettype_               vecgettype
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL vecsettype_(Vec *x,CHAR type_name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = VecSetType(*x,t);
  FREECHAR(type_name,t);
}

void PETSC_STDCALL vecgettype_(Vec *vv,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = VecGetType(*vv,&tname);
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

EXTERN_C_END
