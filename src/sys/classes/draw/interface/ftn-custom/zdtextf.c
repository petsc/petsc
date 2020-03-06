#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawstring_          PETSCDRAWSTRING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawstring_          petscdrawstring
#endif

PETSC_EXTERN void petscdrawstring_(PetscDraw *ctx,double* xl,double* yl,int* cl,char* text,
                                    PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawString(*ctx,*xl,*yl,*cl,t);if (*ierr) return;
  FREECHAR(text,t);
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawstringvertical_  PETSCDRAWSTRINGVERTICAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawstringvertical_  petscdrawstringvertical
#endif

PETSC_EXTERN void petscdrawstringvertical_(PetscDraw *ctx,double *xl,double *yl,int *cl,
                   char* text,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawStringVertical(*ctx,*xl,*yl,*cl,t);if (*ierr) return;
  FREECHAR(text,t);
}

