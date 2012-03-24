#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawopenx_           PETSCDRAWOPENX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawopenx_           petscdrawopenx
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscdrawopenx_(MPI_Comm *comm,CHAR display PETSC_MIXED_LEN(len1),
                    CHAR title PETSC_MIXED_LEN(len2),int *x,int *y,int *w,int *h,PetscDraw* inctx,
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *ierr = PetscDrawOpenX(MPI_Comm_f2c(*(MPI_Fint *)&*comm),t1,t2,*x,*y,*w,*h,inctx);
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}

EXTERN_C_END
