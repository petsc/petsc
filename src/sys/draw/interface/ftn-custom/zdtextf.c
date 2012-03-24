#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawstring_          PETSCDRAWSTRING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawstring_          petscdrawstring
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscdrawstring_(PetscDraw *ctx,double* xl,double* yl,int* cl,CHAR text PETSC_MIXED_LEN(len),
               PetscErrorCode *ierr PETSC_END_LEN(len)){
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawString(*ctx,*xl,*yl,*cl,t);
  FREECHAR(text,t);
}

EXTERN_C_END
