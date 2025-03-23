#include <petsc/private/ftnimpl.h>
#include <petscdraw.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petsc_viewer_draw_         PETSC_VIEWER_DRAW
  #define petscviewermonitorlgsetup_ PETSCVIEWERMONITORLGSETUP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petsc_viewer_draw_         petsc_viewer_draw
  #define petscviewermonitorlgsetup_ petscviewermonitorlgsetup
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
  #define petsc_viewer_draw_ petsc_viewer_draw__
#endif

PETSC_EXTERN PetscViewer petsc_viewer_draw_(MPI_Comm *comm)
{
  return PETSC_VIEWER_DRAW_(MPI_Comm_f2c(*(MPI_Fint *)&*comm));
}

PETSC_EXTERN void petscviewermonitorlgsetup_(PetscViewer *v, char *host, char *label, char *metric, int l, const char **names, int *x, int *y, int *m, int *n, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2, PETSC_FORTRAN_CHARLEN_T len3)
{
  char *t1, *t2, *t3;

  FIXCHAR(host, len1, t1);
  FIXCHAR(label, len2, t2);
  FIXCHAR(metric, len3, t3);
  *ierr = PetscViewerMonitorLGSetUp(*v, t1, t2, t3, l, names, *x, *y, *m, *n);
}
