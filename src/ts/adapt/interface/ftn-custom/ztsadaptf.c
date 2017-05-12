#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tsadaptsettype_ TSADAPTSETTYPE
#define tsadaptgettype_ TSADAPTGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsadaptsettype_ tsadaptsettype
#define tsadaptgettype_ tsadaptgettype
#endif

PETSC_EXTERN void PETSC_STDCALL tsadaptsettype_(TSAdapt *tsadapt,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = TSAdaptSetType(*tsadapt,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL tsadaptgettype_(TSAdapt *adapt,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *t;

  *ierr = TSAdaptGetType(*adapt,&t);
  *ierr = PetscStrncpy(type,t,len);
  FIXRETURNCHAR(PETSC_TRUE,type,len);
}
