#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectsetoptionsprefix     PETSCOBJECTSETOPTIONSPREFIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectsetoptionsprefix_    petscobjectsetoptionsprefix
#endif

PETSC_EXTERN void PETSC_STDCALL petscobjectsetoptionsprefix_(PetscObject *obj,char* prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PetscObjectSetOptionsPrefix(*obj,t);
  FREECHAR(prefix,t);
}



