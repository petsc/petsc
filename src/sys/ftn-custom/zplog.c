#include "private/fortranimpl.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsclogeventbegin_       PETSCLOGEVENTBEGIN
#define petsclogeventend_         PETSCLOGEVENTEND
#define petsclogflops_            PETSCLOGFLOPS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsclogeventbegin_       petsclogeventbegin
#define petsclogeventend_         petsclogeventend
#define petsclogflops_            petsclogflops
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petsclogeventbegin_(PetscLogEvent *e,PetscErrorCode *ierr){
  *ierr = PetscLogEventBegin(*e,0,0,0,0);
}

void PETSC_STDCALL petsclogeventend_(PetscLogEvent *e,PetscErrorCode *ierr){
  *ierr = PetscLogEventEnd(*e,0,0,0,0);
}

void PETSC_STDCALL petsclogflops_(PetscLogDouble *f,PetscErrorCode *ierr) {
  *ierr = PetscLogFlops(*f);
}


EXTERN_C_END
