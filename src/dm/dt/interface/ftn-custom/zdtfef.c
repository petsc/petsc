#include <petsc/private/fortranimpl.h>
#include <petscfe.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscfeview_          PETSCFEVIEW
#define petscfecreatedefault_ PETSCFECREATEDEFAULT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscfeview_          petscfeview
#define petscfecreatedefault_ petscfecreatedefault
#endif

PETSC_EXTERN void petscfeview_(PetscFE *fe,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscFEView(*fe,v);
}

PETSC_EXTERN void petscfecreatedefault_(MPI_Fint *comm,PetscInt *dim,PetscInt *Nc,PetscBool *isSimplex,char* prefix,PetscInt *qorder, PetscFE *fe,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *tprefix;

  FIXCHAR(prefix,len,tprefix);
  *ierr = PetscFECreateDefault(MPI_Comm_f2c(*comm), *dim, *Nc, *isSimplex, tprefix, *qorder, fe);if (*ierr) return;
  FREECHAR(prefix,tprefix);
}
