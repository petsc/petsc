#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petsc_viewer_draw__       PETSC_VIEWER_DRAW_BROKEN
  #define petscviewerdrawgetdraw_   PETSCVIEWERDRAWGETDRAW
  #define petscviewerdrawgetdrawlg_ PETSCVIEWERDRAWGETDRAWLG
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petsc_viewer_draw__       petsc_viewer_draw_
  #define petscviewerdrawgetdraw_   petscviewerdrawgetdraw
  #define petscviewerdrawgetdrawlg_ petscviewerdrawgetdrawlg
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
  #define petsc_viewer_draw__ petsc_viewer_draw___
#endif

PETSC_EXTERN PetscViewer petsc_viewer_draw__(MPI_Comm *comm)
{
  return PETSC_VIEWER_DRAW_(MPI_Comm_f2c(*(MPI_Fint *)&*comm));
}

PETSC_EXTERN void petscviewerdrawgetdraw_(PetscViewer *vin, int *win, PetscDraw *draw, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerDrawGetDraw(v, *win, draw);
}

PETSC_EXTERN void petscviewerdrawgetdrawlg_(PetscViewer *vin, int *win, PetscDrawLG *drawlg, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerDrawGetDrawLG(v, *win, drawlg);
}
