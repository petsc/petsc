#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawaxissetlabels_   PETSCDRAWAXISSETLABELS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawaxissetlabels_   petscdrawaxissetlabels
#endif

PETSC_EXTERN void petscdrawaxissetlabels_(PetscDrawAxis *axis,char* top,
                    char* xlabel,char* ylabel,
                    PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len1,PETSC_FORTRAN_CHARLEN_T len2,PETSC_FORTRAN_CHARLEN_T len3)
{
  char *t1,*t2,*t3;

  FIXCHAR(top,len1,t1);
  FIXCHAR(xlabel,len2,t2);
  FIXCHAR(ylabel,len3,t3);
  *ierr = PetscDrawAxisSetLabels(*axis,t1,t2,t3);if (*ierr) return;
  FREECHAR(top,t1);
  FREECHAR(xlabel,t2);
  FREECHAR(ylabel,t3);
}

