#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawsettype_          PETSCDRAWSETTYPE
#define petscdrawcreate_           PETSCDRAWCREATE
#define petscdrawsetoptionsprefix_ PETSCDRAWSETOPTIONSPREFIX
#define petscdrawviewfromoptions_  PETSCDRAWVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawsettype_          petscdrawsettype
#define petscdrawcreate_           petscdrawcreate
#define petscdrawsetoptionsprefix_ petscdrawsetoptionsprefix
#define petscdrawviewfromoptions_  petscdrawviewfromoptions
#endif

PETSC_EXTERN void PETSC_STDCALL petscdrawsettype_(PetscDraw *ctx,char* text PETSC_MIXED_LEN(len),
                                     PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawSetType(*ctx,t);if (*ierr) return;
  FREECHAR(text,t);
}

PETSC_EXTERN void PETSC_STDCALL petscdrawcreate_(MPI_Comm *comm,char* display PETSC_MIXED_LEN(len1),
                    char* title PETSC_MIXED_LEN(len2),int *x,int *y,int *w,int *h,PetscDraw* inctx,
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *ierr = PetscDrawCreate(MPI_Comm_f2c(*(MPI_Fint*)&*comm),t1,t2,*x,*y,*w,*h,inctx);if (*ierr) return;
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}

PETSC_EXTERN void PETSC_STDCALL petscdrawsetoptionsprefix_(PetscDraw *ctx,char* text PETSC_MIXED_LEN(len),
                                     PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawSetOptionsPrefix(*ctx,t);if (*ierr) return;
  FREECHAR(text,t);
}

PETSC_EXTERN void PETSC_STDCALL petscdrawviewfromoptions_(PetscDraw *draw,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscDrawViewFromOptions(*draw,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

