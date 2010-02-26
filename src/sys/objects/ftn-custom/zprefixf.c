#include "private/fortranimpl.h" 

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscobjectsetoptionsprefix     PETSCOBJECTSETOPTIONSPREFIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectsetoptionsprefix_    petscobjectsetoptionsprefix
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscobjectsetoptionsprefix_(PetscObject *obj,CHAR prefix PETSC_MIXED_LEN(len),
                                        PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PetscObjectSetOptionsPrefix(*obj,t);
  FREECHAR(prefix,t);
}

EXTERN_C_END

