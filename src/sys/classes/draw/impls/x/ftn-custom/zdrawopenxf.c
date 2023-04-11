#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscdrawopenx_ PETSCDRAWOPENX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscdrawopenx_ petscdrawopenx
#endif

PETSC_EXTERN void petscdrawopenx_(MPI_Comm *comm, char *display, char *title, int *x, int *y, int *w, int *h, PetscDraw *inctx, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char *t1, *t2;

  FIXCHAR(display, len1, t1);
  FIXCHAR(title, len2, t2);
  *ierr = PetscDrawOpenX(MPI_Comm_f2c(*(MPI_Fint *)&*comm), t1, t2, *x, *y, *w, *h, inctx);
  if (*ierr) return;
  FREECHAR(display, t1);
  FREECHAR(title, t2);
}
