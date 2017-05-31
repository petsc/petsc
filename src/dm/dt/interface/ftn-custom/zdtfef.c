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

PETSC_EXTERN void PETSC_STDCALL petscfeview_(PetscFE *fe,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscFEView(*fe,v);
}

PETSC_EXTERN void PETSC_STDCALL petscfecreatedefault_(DM *dm,PetscInt *dim,PetscInt *Nc,PetscBool *isSimplex,char* prefix PETSC_MIXED_LEN(len),PetscInt *qorder, PetscFE *fe,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tprefix;

  FIXCHAR(prefix,len,tprefix);
  *ierr = PetscFECreateDefault(*dm, *dim, *Nc, *isSimplex, tprefix, *qorder, fe);
  FREECHAR(prefix,tprefix);
}
