#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawaxissetlabels_   PETSCDRAWAXISSETLABELS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawaxissetlabels_   petscdrawaxissetlabels
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscdrawaxissetlabels_(PetscDrawAxis *axis,CHAR top PETSC_MIXED_LEN(len1),
                    CHAR xlabel PETSC_MIXED_LEN(len2),CHAR ylabel PETSC_MIXED_LEN(len3),
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

EXTERN_C_END
