#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petsclogview_       PETSCLOGVIEW
  #define petsclogeventbegin_ PETSCLOGEVENTBEGIN
  #define petsclogeventend_   PETSCLOGEVENTEND
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petsclogview_       petsclogview
  #define petsclogeventbegin_ petsclogeventbegin
  #define petsclogeventend_   petsclogeventend
#endif

PETSC_EXTERN void petsclogeventbegin_(PetscLogEvent *e, PetscErrorCode *ierr)
{
  *ierr = PetscLogEventBegin(*e, 0, 0, 0, 0);
}

PETSC_EXTERN void petsclogeventend_(PetscLogEvent *e, PetscErrorCode *ierr)
{
  *ierr = PetscLogEventEnd(*e, 0, 0, 0, 0);
}

PETSC_EXTERN void petsclogview_(PetscViewer *viewer, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
#if defined(PETSC_USE_LOG)
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscLogView(v);
#endif
}
