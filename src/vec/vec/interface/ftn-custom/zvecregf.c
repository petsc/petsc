#include <petsc/private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecsettype_               VECSETTYPE
#define vecgettype_               VECGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecsettype_               vecsettype
#define vecgettype_               vecgettype
#endif

PETSC_EXTERN void vecsettype_(Vec *x,char* type_name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = VecSetType(*x,t);if (*ierr) return;
  FREECHAR(type_name,t);
}

PETSC_EXTERN void vecgettype_(Vec *vv,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = VecGetType(*vv,&tname);
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}
