#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawgettitle_        PETSCDRAWGETTITLE
#define petscdrawsettitle_        PETSCDRAWSETTITLE
#define petscdrawappendtitle_     PETSCDRAWAPPENDTITLE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawgettitle_        petscdrawgettitle
#define petscdrawsettitle_        petscdrawsettitle
#define petscdrawappendtitle_     petscdrawappendtitle
#endif

PETSC_EXTERN void PETSC_STDCALL petscdrawgettitle_(PetscDraw *draw,CHAR title PETSC_MIXED_LEN(len),
                                      PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c3,*t;
  int  len3;
  c3    = title;
  len3  = len - 1;
  *ierr = PetscDrawGetTitle(*draw,&t);
  *ierr = PetscStrncpy(c3,t,len3);
}

PETSC_EXTERN void PETSC_STDCALL petscdrawsettitle_(PetscDraw *draw,CHAR title PETSC_MIXED_LEN(len),
                                      PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t1;
  FIXCHAR(title,len,t1);
  *ierr = PetscDrawSetTitle(*draw,t1);
  FREECHAR(title,t1);
}

PETSC_EXTERN void PETSC_STDCALL petscdrawappendtitle_(PetscDraw *draw,CHAR title PETSC_MIXED_LEN(len),
                                         PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t1;
  FIXCHAR(title,len,t1);
  *ierr = PetscDrawAppendTitle(*draw,t1);
  FREECHAR(title,t1);
}

