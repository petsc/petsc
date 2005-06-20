#include "zpetsc.h"
#include "petsc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewersocketputscalar_ PETSCVIEWERSOCKETPUTSCALAR
#define petscviewersocketputint_    PETSCVIEWERSOCKETPUTINT
#define petscviewersocketputreal_   PETSCVIEWERSOCKETPUTREAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewersocketputscalar_ petscviewersocketputscalar
#define petscviewersocketputint_    petscviewersocketputint
#define petscviewersocketputreal_   petscviewersocketputreal
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscviewersocketputscalar_(PetscViewer *viewer,PetscInt *m,PetscInt *n,PetscScalar *s,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSocketPutScalar(v,*m,*n,s);
}

void PETSC_STDCALL petscviewersocketputreal_(PetscViewer *viewer,PetscInt *m,PetscInt *n,PetscReal *s,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSocketPutReal(v,*m,*n,s);
}

void PETSC_STDCALL petscviewersocketputint_(PetscViewer *viewer,PetscInt *m,PetscInt *s,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSocketPutInt(v,*m,s);
}

EXTERN_C_END
