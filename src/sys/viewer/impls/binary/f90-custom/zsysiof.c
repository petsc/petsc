#include "src/sys/f90/f90impl.h"
#include "zpetsc.h"
#include "petscviewer.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerbinarywriteint_ PETSCVIEWERBINARYWRITEINT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerbinarywriteint_ petscviewerbinarywriteint
#endif

EXTERN_C_BEGIN


void PETSC_STDCALL petscviewerbinarywriteint_(PetscViewer *viewer,PetscInt *a,PetscInt *len,PetscTruth *tmp,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryWrite(v,a,*len,PETSC_INT,*tmp);
}


EXTERN_C_END
