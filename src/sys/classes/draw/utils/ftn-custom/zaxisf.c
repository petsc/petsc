#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawaxissetlabels_   PETSCDRAWAXISSETLABELS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawaxissetlabels_   petscdrawaxissetlabels
#endif

PETSC_EXTERN void PETSC_STDCALL petscdrawaxissetlabels_(PetscDrawAxis *axis,char* top PETSC_MIXED_LEN(len1),
                    char* xlabel PETSC_MIXED_LEN(len2),char* ylabel PETSC_MIXED_LEN(len3),
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len3))
{
  char *t1,*t2,*t3;

  FIXCHAR(top,len1,t1);
  FIXCHAR(xlabel,len2,t2);
  FIXCHAR(ylabel,len3,t3);
  *ierr = PetscDrawAxisSetLabels(*axis,t1,t2,t3);
  FREECHAR(top,t1);
  FREECHAR(xlabel,t2);
  FREECHAR(ylabel,t3);
}

