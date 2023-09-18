#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscdrawgettitle_          PETSCDRAWGETTITLE
  #define petscdrawsettitle_          PETSCDRAWSETTITLE
  #define petscdrawappendtitle_       PETSCDRAWAPPENDTITLE
  #define petscdrawsetsavefinalimage_ PETSCDRAWSETSAVEFINALIMAGE
  #define petscdrawsetsave_           PETSCDRAWSETSAVE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscdrawgettitle_          petscdrawgettitle
  #define petscdrawsettitle_          petscdrawsettitle
  #define petscdrawappendtitle_       petscdrawappendtitle
  #define petscdrawsetsavefinalimage_ petscdrawsetsavefinalimage
  #define petscdrawsetsave_           petscdrawsetsave
#endif

PETSC_EXTERN void petscdrawgettitle_(PetscDraw *draw, char *title, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *t;
  *ierr = PetscDrawGetTitle(*draw, &t);
  *ierr = PetscStrncpy(title, t, len);
  FIXRETURNCHAR(PETSC_TRUE, title, len);
}

PETSC_EXTERN void petscdrawsettitle_(PetscDraw *draw, char *title, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t1;
  FIXCHAR(title, len, t1);
  *ierr = PetscDrawSetTitle(*draw, t1);
  if (*ierr) return;
  FREECHAR(title, t1);
}

PETSC_EXTERN void petscdrawappendtitle_(PetscDraw *draw, char *title, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t1;
  FIXCHAR(title, len, t1);
  *ierr = PetscDrawAppendTitle(*draw, t1);
  if (*ierr) return;
  FREECHAR(title, t1);
}

PETSC_EXTERN void petscdrawsetsavefinalimage_(PetscDraw *draw, char *filename, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t1;
  FIXCHAR(filename, len, t1);
  *ierr = PetscDrawSetSaveFinalImage(*draw, t1);
  if (*ierr) return;
  FREECHAR(filename, t1);
}

PETSC_EXTERN void petscdrawsetsave_(PetscDraw *draw, char *filename, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t1;
  FIXCHAR(filename, len, t1);
  *ierr = PetscDrawSetSave(*draw, t1);
  if (*ierr) return;
  FREECHAR(filename, t1);
}
