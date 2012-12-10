#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawsettype_         PETSCDRAWSETTYPE
#define petscdrawcreate_          PETSCDRAWCREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawsettype_         petscdrawsettype
#define petscdrawcreate_          petscdrawcreate
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscdrawsettype_(PetscDraw *ctx,CHAR text PETSC_MIXED_LEN(len),
               PetscErrorCode *ierr PETSC_END_LEN(len)){
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawSetType(*ctx,t);
  FREECHAR(text,t);
}

void PETSC_STDCALL petscdrawcreate_(MPI_Comm *comm,CHAR display PETSC_MIXED_LEN(len1),
                    CHAR title PETSC_MIXED_LEN(len2),int *x,int *y,int *w,int *h,PetscDraw* inctx,
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *ierr = PetscDrawCreate(MPI_Comm_f2c(*(MPI_Fint *)&*comm),t1,t2,*x,*y,*w,*h,inctx);
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}

EXTERN_C_END
